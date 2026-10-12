"""``resolve_mcp_servers`` and the room ownership around session setup.

Behavior over a real ACP wire (``FakeACPAgent`` in process): what the agent
receives, when the callback runs, and which resources a room keeps until its
agent process is confirmed closed.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from acp import RequestError
from acp.schema import (
    AcpMcpServer,
    EnvVariable,
    HttpHeader,
    HttpMcpServer,
    McpServerStdio,
    SseMcpServer,
)

from band.integrations.acp.client_adapter import (
    ACPMcpServerResolutionError,
    ResolvedMcpServer,
)
from band.integrations.acp.client_runtime import ACPProcessCloseError
from band.integrations.acp.client_types import ACPClientSessionState
from band.runtime.tools import BAND_MCP_SERVER_NAME
from tests.integrations.acp.acp_toolkit import (
    DEFAULT_ROOM,
    FakeACPAgent,
    acp_adapter,
    fake_agent_config,
)

SECRET = "s3cret-token"


def http_server(
    name: str = "repo", url: str = "http://127.0.0.1:9/mcp"
) -> HttpMcpServer:
    return HttpMcpServer(
        type="http",
        name=name,
        url=url,
        headers=[HttpHeader(name="Authorization", value=f"Bearer {SECRET}")],
    )


def sse_server(name: str = "events") -> SseMcpServer:
    return SseMcpServer(
        type="sse",
        name=name,
        url="http://127.0.0.1:9/sse",
        headers=[HttpHeader(name="Authorization", value=f"Bearer {SECRET}")],
    )


def stdio_server(name: str = "tools") -> McpServerStdio:
    return McpServerStdio(
        name=name,
        command="/bin/echo",
        args=["mcp"],
        env=[EnvVariable(name="TOKEN", value=SECRET)],
    )


class Resolver:
    """Records each call's workspace and returns the scripted servers."""

    def __init__(self, *servers: Any) -> None:
        self.servers: list[Any] = list(servers)
        self.workspaces: list[str] = []

    async def __call__(self, workspace: str) -> Sequence[ResolvedMcpServer]:
        self.workspaces.append(workspace)
        return self.servers


def server_names(servers: list[Any]) -> list[str]:
    return [server.name for server in servers]


async def spawn_sleeper() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)"
    )


def failing_wait_once(process: asyncio.subprocess.Process) -> None:
    """Make the first confirmation of ``process`` fail, as a reap that errors."""
    real_wait = process.wait
    calls = 0

    async def wait() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("wait failed")
        return await real_wait()

    process.wait = wait  # type: ignore[method-assign]


async def test_resolved_servers_replace_configured_ones_once_per_session() -> None:
    agent = FakeACPAgent(http=True)
    resolver = Resolver(http_server(), stdio_server())
    config = fake_agent_config(mcp_servers=[http_server(name="static")])

    async with acp_adapter(agent, config, resolve_mcp_servers=resolver) as acp:
        await acp.send("one")
        await acp.send("two")
        workspace = acp.adapter._workspaces.workspace(DEFAULT_ROOM)

    assert resolver.workspaces == [workspace]
    assert len(agent.sessions) == 1
    sent = agent.sessions[0]["mcp_servers"]
    assert server_names(sent) == ["repo", "tools"]
    assert sent[0].headers[0].value == f"Bearer {SECRET}"
    assert sent[1].env[0].value == SECRET


async def test_empty_result_is_authoritative() -> None:
    agent = FakeACPAgent(http=True)
    config = fake_agent_config(mcp_servers=[http_server(name="static")])

    async with acp_adapter(agent, config, resolve_mcp_servers=Resolver()) as acp:
        await acp.send("hello")

    assert agent.sessions[0]["mcp_servers"] == []


async def test_failed_resolution_never_falls_back_to_configured_servers() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    config = fake_agent_config(mcp_servers=[http_server(name="static")])
    resolver = Resolver(http_server(name="fixed"))
    fail = True

    async def flaky(workspace: str) -> Sequence[ResolvedMcpServer]:
        if fail:
            raise LookupError("grant unavailable")
        return await resolver(workspace)

    async with acp_adapter(agent, config, resolve_mcp_servers=flaky) as acp:
        with pytest.raises(LookupError):
            await acp.send("first")
        assert agent.sessions == []
        fail = False
        await acp.send("second")

    assert [server_names(s["mcp_servers"]) for s in agent.sessions] == [["fixed"]]


@pytest.mark.parametrize("resolved", [[], [stdio_server()]], ids=["empty", "one"])
async def test_configured_band_routing_is_kept(resolved: list[Any]) -> None:
    agent = FakeACPAgent(http=True)
    band = http_server(name=BAND_MCP_SERVER_NAME, url="http://127.0.0.1:9/band")
    config = fake_agent_config(mcp_servers=[http_server(name="static"), band])

    async with acp_adapter(
        agent, config, resolve_mcp_servers=Resolver(*resolved)
    ) as acp:
        await acp.send("hello")

    assert server_names(agent.sessions[0]["mcp_servers"]) == [
        *server_names(resolved),
        BAND_MCP_SERVER_NAME,
    ]


async def test_band_server_follows_resolved_servers() -> None:
    agent = FakeACPAgent(http=True)

    async with acp_adapter(
        agent,
        fake_agent_config(inject_band_tools=True),
        resolve_mcp_servers=Resolver(http_server()),
    ) as acp:
        await acp.send("hello")

    assert server_names(agent.sessions[0]["mcp_servers"]) == [
        "repo",
        BAND_MCP_SERVER_NAME,
    ]


async def test_each_fresh_connection_resolves_anew() -> None:
    agent = FakeACPAgent(http=True)
    resolver = Resolver(http_server(url="http://127.0.0.1:9/grant-a"))

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        await acp.send("one")
        await acp.adapter.on_cleanup(DEFAULT_ROOM)
        resolver.servers = [http_server(url="http://127.0.0.1:9/grant-b")]
        await acp.send("two")
        await acp.send("three")

    assert len(resolver.workspaces) == 2
    assert [s["mcp_servers"][0].url for s in agent.sessions] == [
        "http://127.0.0.1:9/grant-a",
        "http://127.0.0.1:9/grant-b",
    ]


async def test_concurrent_turns_share_one_resolution_and_rooms_are_independent() -> (
    None
):
    agent = FakeACPAgent(http=True)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        calls.append(workspace)
        if workspace.endswith(DEFAULT_ROOM):
            entered.set()
            await release.wait()
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        first = asyncio.create_task(acp.send("a"))
        second = asyncio.create_task(acp.send("b"))
        await entered.wait()

        await acp.send("other room", room="room-2")
        assert not first.done() and not second.done()

        release.set()
        await asyncio.gather(first, second)

    assert len(calls) == 2
    assert len(agent.sessions) == 2


async def test_one_waiter_leaving_does_not_cancel_the_shared_setup() -> None:
    agent = FakeACPAgent(http=True)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        leaving = asyncio.create_task(acp.send("a"))
        staying = asyncio.create_task(acp.send("b"))
        await entered.wait()
        leaving.cancel()
        await asyncio.gather(leaving, return_exceptions=True)

        release.set()
        await staying

    assert calls == 1
    assert len(agent.sessions) == 1


async def test_last_waiter_leaving_cancels_setup_before_any_session() -> None:
    agent = FakeACPAgent(http=True)
    entered = asyncio.Event()

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        entered.set()
        await asyncio.Event().wait()
        return []

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        turn = asyncio.create_task(acp.send("a"))
        await entered.wait()
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)

        assert acp.adapter._session_initializers == {}
        assert agent.sessions == []


async def test_setup_retiring_its_own_room_does_not_deadlock() -> None:
    agent = FakeACPAgent(http=True)
    acp_ref: list[Any] = []

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        await acp_ref[0].adapter.on_cleanup(DEFAULT_ROOM)
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        acp_ref.append(acp)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(acp.send("a"), timeout=5)
        await asyncio.sleep(0.05)

        assert agent.sessions == []
        assert DEFAULT_ROOM not in acp.adapter._runtimes
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms


async def test_load_and_fresh_fallback_share_one_stable_snapshot() -> None:
    agent = FakeACPAgent(http=True, supports_session_load=True)
    returned = http_server(url="http://127.0.0.1:9/original")
    resolver = Resolver(returned)
    loaded_with: list[Any] = []
    unknown = agent.load_session

    async def load_session(
        cwd: str, session_id: str, mcp_servers: Any = None, **kwargs: Any
    ) -> Any:
        loaded_with.extend(mcp_servers or [])
        # The callback's own objects change after it returned them.
        returned.url = "http://127.0.0.1:9/changed"
        return await unknown(cwd, session_id, mcp_servers, **kwargs)

    agent.load_session = load_session  # type: ignore[method-assign]
    history = ACPClientSessionState(room_to_session={DEFAULT_ROOM: "gone"})

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        await acp.send("hello", history=history, bootstrap=True)

    assert len(resolver.workspaces) == 1
    assert [s.url for s in loaded_with] == ["http://127.0.0.1:9/original"]
    assert agent.sessions[0]["mcp_servers"] == loaded_with


@pytest.mark.parametrize(
    ("http", "sse", "server", "transport"),
    [
        # Neither advertised: Band still prefers HTTP, which is no evidence.
        (False, False, http_server(), "http"),
        (False, True, http_server(), "http"),
        (True, False, sse_server(), "sse"),
    ],
)
async def test_unadvertised_remote_transport_is_refused_without_secrets(
    http: bool, sse: bool, server: Any, transport: str
) -> None:
    agent = FakeACPAgent(http=http, sse=sse)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=Resolver(server)
    ) as acp:
        with pytest.raises(ACPMcpServerResolutionError) as caught:
            await acp.send("hello")

    message = str(caught.value)
    assert server.name in message and transport in message
    assert SECRET not in message and server.url not in message
    assert agent.sessions == []


async def test_advertised_transports_and_stdio_are_accepted() -> None:
    agent = FakeACPAgent(http=True, sse=True)
    resolver = Resolver(http_server(), sse_server(), stdio_server())

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        await acp.send("hello")

    assert server_names(agent.sessions[0]["mcp_servers"]) == [
        "repo",
        "events",
        "tools",
    ]


@pytest.mark.parametrize(
    "servers",
    [
        [http_server(name="repo"), stdio_server(name="repo")],
        [http_server(name=BAND_MCP_SERVER_NAME)],
        [AcpMcpServer(type="acp", name="peer", server_id="peer-1")],
    ],
    ids=["duplicate", "band", "acp-transport"],
)
async def test_unacceptable_servers_fail_setup(servers: list[Any]) -> None:
    agent = FakeACPAgent(http=True)
    config = fake_agent_config(mcp_servers=[http_server(name="static")])

    async with acp_adapter(
        agent, config, resolve_mcp_servers=Resolver(*servers)
    ) as acp:
        with pytest.raises(ACPMcpServerResolutionError):
            await acp.send("hello")

    assert agent.sessions == []


@pytest.mark.parametrize("returned", ["repo", [{"name": "repo"}]])
async def test_non_descriptor_results_are_type_errors(returned: Any) -> None:
    async def resolver(workspace: str) -> Any:
        return returned

    agent = FakeACPAgent(http=True)
    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        with pytest.raises(TypeError):
            await acp.send("hello")

    assert agent.sessions == []


async def test_setup_failure_survives_a_failed_close_and_next_turn_recovers() -> None:
    agent = FakeACPAgent(http=True).will_say("recovered")
    sleeper = await spawn_sleeper()
    failing_wait_once(sleeper)
    calls = 0
    acp_ref: list[Any] = []

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        nonlocal calls
        calls += 1
        if calls == 1:
            # The room's agent process, as one that will not confirm closed.
            acp_ref[0].adapter._runtimes[DEFAULT_ROOM]._process = sleeper
            raise LookupError("resolver failed")
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        acp_ref.append(acp)
        with pytest.raises(LookupError):
            await acp.send("first")
        failed = acp.adapter._runtimes[DEFAULT_ROOM]
        assert failed.close_pending
        assert DEFAULT_ROOM in acp.adapter._workspaces.rooms

        reply = await acp.send("second")

        assert reply.thoughts == ["recovered"]
        assert acp.adapter._runtimes[DEFAULT_ROOM] is not failed
    assert sleeper.returncode is not None


async def test_cleanup_keeps_an_unconfirmed_runtime_owned_for_retry() -> None:
    agent = FakeACPAgent(http=True).will_say("hi")
    sleeper = await spawn_sleeper()
    failing_wait_once(sleeper)

    async with acp_adapter(agent) as acp:
        await acp.send("hello")
        runtime = acp.adapter._runtimes[DEFAULT_ROOM]
        runtime._process = sleeper

        with pytest.raises(ACPProcessCloseError):
            await acp.adapter.on_cleanup(DEFAULT_ROOM)
        assert acp.adapter._runtimes[DEFAULT_ROOM] is runtime
        assert DEFAULT_ROOM in acp.adapter._workspaces.rooms

        await acp.adapter.on_cleanup(DEFAULT_ROOM)

        assert DEFAULT_ROOM not in acp.adapter._runtimes
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms
    assert sleeper.returncode is not None


async def test_cleanup_waits_for_setup_to_settle_and_nothing_is_published() -> None:
    agent = FakeACPAgent(http=True)
    entered = asyncio.Event()
    settle = asyncio.Event()

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        entered.set()
        # A callback that swallows the cancel, finishes its own work, and
        # returns a result anyway.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await settle.wait()
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        turn = asyncio.create_task(acp.send("hello"))
        await entered.wait()
        cleanup = asyncio.create_task(acp.adapter.on_cleanup(DEFAULT_ROOM))
        await asyncio.sleep(0.05)

        # Unsettled setup: the room still owns its runtime and workspace.
        assert not cleanup.done()
        assert DEFAULT_ROOM in acp.adapter._runtimes
        assert DEFAULT_ROOM in acp.adapter._workspaces.rooms

        settle.set()
        await cleanup
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert agent.sessions == []
        assert DEFAULT_ROOM not in acp.adapter._room_to_session
        assert DEFAULT_ROOM not in acp.adapter._runtimes
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms


async def test_cancelled_cleanup_still_finishes_before_the_room_respawns() -> None:
    agent = FakeACPAgent(http=True).will_say("hi")
    spawned: list[object] = []

    async with acp_adapter(agent) as acp:
        await acp.send("hello")
        old = acp.adapter._runtimes[DEFAULT_ROOM]
        build = acp.adapter._build_runtime

        def counting_build(workspace: str | None = None) -> Any:
            runtime = build(workspace)
            spawned.append(runtime)
            return runtime

        acp.adapter._build_runtime = counting_build  # type: ignore[method-assign]
        gate = agent.exits_slowly()
        cleanup = asyncio.create_task(acp.adapter.on_cleanup(DEFAULT_ROOM))
        await gate.received.wait()
        cleanup.cancel()
        next_turn = asyncio.create_task(acp.send("again"))
        await asyncio.sleep(0.05)

        # The old process is still exiting: no second process for the room yet.
        assert spawned == []
        assert acp.adapter._runtimes[DEFAULT_ROOM] is old

        agent.exit_gate = None
        gate.release.set()
        reply = await next_turn

        assert reply.thoughts == ["hi"]
        assert acp.adapter._runtimes[DEFAULT_ROOM] is spawned[0]
    with pytest.raises(asyncio.CancelledError):
        await cleanup


async def test_resolver_error_does_not_leak_into_static_rooms() -> None:
    """Without a resolver nothing about session MCP setup changes."""
    agent = FakeACPAgent(http=False, sse=False).will_say("ok")
    config = fake_agent_config(mcp_servers=[http_server(name="static")])

    async with acp_adapter(agent, config) as acp:
        reply = await acp.send("hello")

    assert reply.thoughts == ["ok"]
    assert server_names(agent.sessions[0]["mcp_servers"]) == ["static"]


async def test_wedged_setup_error_is_kept_when_retirement_close_fails() -> None:
    agent = FakeACPAgent(http=True).wedges_its_process()
    sleeper = await spawn_sleeper()
    failing_wait_once(sleeper)
    acp_ref: list[Any] = []

    async def resolver(workspace: str) -> list[ResolvedMcpServer]:
        acp_ref[0].adapter._runtimes[DEFAULT_ROOM]._process = sleeper
        return []

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        acp_ref.append(acp)
        with pytest.raises(RequestError):
            await acp.send("hello")
        assert acp.adapter._runtimes[DEFAULT_ROOM].close_pending
    assert sleeper.returncode is not None
