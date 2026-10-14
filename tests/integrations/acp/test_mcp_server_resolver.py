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
    ACPAdapterShutDownError,
    ACPMcpServerResolutionError,
    ResolvedMcpServer,
)
from band.integrations.acp.client_runtime import ACPProcessCloseError
from band.integrations.acp.client_types import ACPClientSessionState
from band.integrations.mcp import BandMCPBackendStoppedError
from band.runtime.tools import BAND_MCP_SERVER_NAME
from tests.conftest import eager_tasks, requires_eager_tasks
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
        await asyncio.sleep(0.01)  # the abandoned setup settles on its own

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
    remote = http_server(url="http://127.0.0.1:9/original")
    local = stdio_server()
    resolver = Resolver(remote, local)
    loaded_with: list[Any] = []
    unknown = agent.load_session

    async def load_session(
        cwd: str, session_id: str, mcp_servers: Any = None, **kwargs: Any
    ) -> Any:
        loaded_with.extend(mcp_servers or [])
        # The callback's own objects change after it returned them, down to
        # nested header, argument and environment values.
        remote.url = "http://127.0.0.1:9/changed"
        remote.headers[0].value = "Bearer rotated"
        local.args.append("--changed")
        local.env[0].value = "rotated"
        return await unknown(cwd, session_id, mcp_servers, **kwargs)

    agent.load_session = load_session  # type: ignore[method-assign]
    history = ACPClientSessionState(room_to_session={DEFAULT_ROOM: "gone"})

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        await acp.send("hello", history=history, bootstrap=True)

    assert len(resolver.workspaces) == 1
    loaded_remote, loaded_local = loaded_with
    assert loaded_remote.url == "http://127.0.0.1:9/original"
    assert loaded_remote.headers[0].value == f"Bearer {SECRET}"
    assert loaded_local.args == ["mcp"]
    assert loaded_local.env[0].value == SECRET
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


def record_spawns(adapter: Any, log: list[str]) -> None:
    """Log ``spawn`` whenever the adapter builds a room runtime."""
    build = adapter._build_runtime

    def building(workspace: str | None = None) -> Any:
        log.append("spawn")
        return build(workspace)

    adapter._build_runtime = building


def record_close(adapter: Any, room: str, log: list[str]) -> None:
    """Log ``close entered`` when ``room``'s current runtime starts closing
    and ``closed`` once it finishes."""
    runtime = adapter._runtimes[room]
    close = runtime.close

    async def closing() -> None:
        log.append("close entered")
        await close()
        log.append("closed")

    runtime.close = closing


def hold_backend_stop(adapter: Any) -> Gate:
    """Hold the running Band MCP backend's stop until the gate is released."""
    backend = adapter._mcp.current
    assert backend is not None
    stop = backend.stop
    gate = Gate()

    async def held() -> None:
        gate.received.set()
        await gate.release.wait()
        await stop()
        gate.finished = True

    object.__setattr__(backend, "stop", held)
    return gate


class Gate:
    def __init__(self) -> None:
        self.received = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False


class HeldCallback:
    """A callback that, once cancelled, finishes its own settlement (until
    ``settle``) and returns a result anyway. Each further cancellation
    interrupts that settlement and is counted."""

    def __init__(self, log: list[str] | None = None) -> None:
        self.entered = asyncio.Event()
        self.settling = asyncio.Event()
        self.settle = asyncio.Event()
        self.log = log if log is not None else []
        self.calls = 0
        self.cancellations = 0

    async def __call__(self, workspace: str) -> list[ResolvedMcpServer]:
        self.calls += 1
        if self.calls > 1:
            self.log.append("second call")
            return [http_server()]
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations += 1
        await self.settlement()
        return [http_server()]

    async def settlement(self) -> None:
        self.settling.set()
        try:
            await self.settle.wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            self.log.append("settlement interrupted")
            raise
        self.log.append("settled")


async def abandon_setup(acp: Any, callback: HeldCallback) -> asyncio.Task[Any]:
    """Start a turn, wait for its callback, then cancel the only waiter and
    wait for it to leave (bounded); its setup is left settling."""
    turn = asyncio.create_task(acp.send("abandoned"))
    await callback.entered.wait()
    turn.cancel()
    await callback.settling.wait()
    async with asyncio.timeout(5):
        await asyncio.gather(turn, return_exceptions=True)
    return turn


def admission_barrier(adapter: Any) -> asyncio.Event:
    """An event set as a turn reaches its room's setup registry (right after
    it has the session's MCP servers, before it joins or starts a setup)."""
    servers = adapter._session_mcp_servers
    reached = asyncio.Event()

    async def reaching(*args: Any) -> Any:
        result = await servers(*args)
        reached.set()
        return result

    adapter._session_mcp_servers = reaching
    return reached


async def test_abandoned_setup_holds_the_room_until_it_settles() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    log: list[str] = []
    callback = HeldCallback(log)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=callback
    ) as acp:
        try:
            await abandon_setup(acp, callback)
            attempted = admission_barrier(acp.adapter)
            follow_up = asyncio.create_task(acp.send("next"))
            # The follow-up reaches the room's setup registry while the
            # abandoned setup is still settling.
            await attempted.wait()
        finally:
            callback.settle.set()
        reply = await follow_up

    assert reply.thoughts == ["ok"]
    # No second setup beside the unsettled one; one cancellation only.
    assert log == ["settled", "second call"]
    assert callback.cancellations == 1
    # The abandoned setup's late result reached neither the agent nor the room.
    assert len(agent.sessions) == 1


async def test_cleanup_joins_an_abandoned_setup_without_cancelling_again() -> None:
    agent = FakeACPAgent(http=True)
    log: list[str] = []
    callback = HeldCallback(log)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=callback
    ) as acp:
        try:
            await abandon_setup(acp, callback)
            record_close(acp.adapter, DEFAULT_ROOM, log)
            retirement_started = enter_retirement_barrier(acp.adapter)
            cleanups = [
                asyncio.create_task(acp.adapter.on_cleanup(DEFAULT_ROOM)),
                asyncio.create_task(acp.adapter.on_cleanup(DEFAULT_ROOM)),
            ]
            await retirement_started.wait()
        finally:
            callback.settle.set()
        await asyncio.gather(*cleanups)

        # The runtime and claim outlived the setup's settlement.
        assert log == ["settled", "close entered", "closed"]
        assert callback.cancellations == 1
        assert agent.sessions == []
        assert DEFAULT_ROOM not in acp.adapter._runtimes
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms


class SelfCancellingCallback(HeldCallback):
    """Cancels its own task, catches it without uncancelling, then settles;
    ``retire_self`` makes it request its room's cleanup from inside setup."""

    def __init__(self, log: list[str], retire_self: Any = None) -> None:
        super().__init__(log)
        self.retire_self = retire_self

    async def __call__(self, workspace: str) -> list[ResolvedMcpServer]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancellations += 1
        if self.retire_self is not None:
            await self.retire_self()
        self.entered.set()
        await self.settlement()
        return [http_server()]


def enter_retirement_barrier(adapter: Any) -> asyncio.Event:
    """An event set once a room retirement's own task starts running."""
    run = adapter._run_retirement
    started = asyncio.Event()

    async def running(*args: Any) -> None:
        started.set()
        await run(*args)

    adapter._run_retirement = running
    return started


async def test_external_cleanup_joins_an_already_cancelling_setup() -> None:
    agent = FakeACPAgent(http=True)
    log: list[str] = []
    callback = SelfCancellingCallback(log)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=callback
    ) as acp:
        turn = asyncio.create_task(acp.send("hello"))
        await callback.settling.wait()
        record_close(acp.adapter, DEFAULT_ROOM, log)
        retirement_started = enter_retirement_barrier(acp.adapter)
        cleanup = asyncio.create_task(acp.adapter.on_cleanup(DEFAULT_ROOM))
        # The retirement has taken the setup over before it may settle.
        await retirement_started.wait()
        callback.settle.set()
        await cleanup
        with pytest.raises(asyncio.CancelledError):
            await turn

        assert log == ["settled", "close entered", "closed"]
        assert callback.cancellations == 1
        assert agent.sessions == []
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms


async def test_self_retirement_joins_an_already_cancelling_setup() -> None:
    agent = FakeACPAgent(http=True)
    log: list[str] = []
    acp_ref: list[Any] = []

    async def retire_self() -> None:
        adapter = acp_ref[0].adapter
        record_close(adapter, DEFAULT_ROOM, log)
        await adapter.on_cleanup(DEFAULT_ROOM)

    callback = SelfCancellingCallback(log, retire_self)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=callback
    ) as acp:
        acp_ref.append(acp)
        turn = asyncio.create_task(acp.send("hello"))
        await callback.settling.wait()
        retirement = acp.adapter._retirements[DEFAULT_ROOM]
        callback.settle.set()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await asyncio.wait((retirement.task,))

        assert log == ["settled", "close entered", "closed"]
        assert callback.cancellations == 1
        assert agent.sessions == []
        assert DEFAULT_ROOM not in acp.adapter._runtimes
        assert DEFAULT_ROOM not in acp.adapter._workspaces.rooms


async def test_a_setup_cancelled_by_anyone_never_reaches_the_agent() -> None:
    agent = FakeACPAgent(http=True)

    async def self_cancelling(workspace: str) -> list[ResolvedMcpServer]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass  # swallowed, never uncancelled
        return [http_server()]

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=self_cancelling
    ) as acp:
        with pytest.raises(asyncio.CancelledError):
            await acp.send("hello")

    assert agent.sessions == []


@pytest.mark.parametrize("inject_band_tools", [False, True])
async def test_final_cleanup_admits_no_room_until_started(
    inject_band_tools: bool,
) -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    config = fake_agent_config(inject_band_tools=inject_band_tools)
    log: list[str] = []

    async with acp_adapter(agent, config) as acp:
        await acp.send("before", room="room-1")
        record_spawns(acp.adapter, log)
        gate = agent.exits_slowly()
        shutdown = asyncio.create_task(acp.adapter.cleanup_all(final=True))
        await gate.received.wait()
        queued = asyncio.create_task(acp.send("queued", room="room-1"))
        new = asyncio.create_task(acp.send("new", room="room-2"))
        gate.release.set()
        await shutdown

        for turn in (queued, new):
            with pytest.raises(ACPAdapterShutDownError):
                await turn
        assert log == []
        assert acp.adapter._runtimes == {}
        assert acp.adapter._workspaces.rooms == ()

        await acp.adapter.on_started("Fake Agent", "restarted")
        reply = await acp.send("after restart", room="room-2")

    assert reply.thoughts == ["ok"]


@pytest.mark.parametrize("inject_band_tools", [False, True])
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_restartable_cleanup_admits_rooms_once_it_settles(
    inject_band_tools: bool, cancel_caller: bool
) -> None:
    """A turn arriving during a restartable cleanup starts only after the old
    process closed, even when the cleanup's caller was cancelled."""
    agent = FakeACPAgent(http=True).will_say("ok")
    config = fake_agent_config(inject_band_tools=inject_band_tools)
    log: list[str] = []

    async with acp_adapter(agent, config) as acp:
        await acp.send("before", room="room-1")
        record_spawns(acp.adapter, log)
        record_close(acp.adapter, "room-1", log)
        gate = agent.exits_slowly()
        cleanup = asyncio.create_task(acp.adapter.cleanup_all(final=False))
        await gate.received.wait()
        if cancel_caller:
            cleanup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cleanup
        new = asyncio.create_task(acp.send("new", room="room-2"))
        gate.release.set()
        reply = await new
        if not cancel_caller:
            await cleanup

    assert reply.thoughts == ["ok"]
    assert log == ["close entered", "closed", "spawn"]


async def test_a_cancelled_joiner_leaves_the_live_joiner_its_cleanup() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    log: list[str] = []

    async with acp_adapter(agent) as acp:
        await acp.send("before", room="room-1")
        record_spawns(acp.adapter, log)
        record_close(acp.adapter, "room-1", log)
        gate = agent.exits_slowly()
        leaving = asyncio.create_task(acp.adapter.cleanup_all(final=False))
        staying = asyncio.create_task(acp.adapter.cleanup_all(final=False))
        await gate.received.wait()
        leaving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leaving
        new = asyncio.create_task(acp.send("new", room="room-2"))
        gate.release.set()
        await staying
        await new

    assert log == ["close entered", "closed", "spawn"]


async def test_a_cancelled_final_caller_still_stops_the_band_server() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")

    async with acp_adapter(agent, fake_agent_config(inject_band_tools=True)) as acp:
        await acp.send("before", room="room-1")
        stopping = hold_backend_stop(acp.adapter)
        caller = asyncio.create_task(acp.adapter.cleanup_all(final=True))
        await stopping.received.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        with pytest.raises(ACPAdapterShutDownError):
            await acp.send("late", room="room-2")
        stopping.release.set()
        await acp.adapter.cleanup_all(final=True)

        assert stopping.finished
        assert acp.adapter._mcp.current is None
        assert acp.adapter._runtimes == {}


async def test_final_cleanup_takes_over_a_restartable_one_and_refuses_parked_turns() -> (
    None
):
    agent = FakeACPAgent(http=True).will_say("ok")
    log: list[str] = []

    async with acp_adapter(agent) as acp:
        await acp.send("before", room="room-1")
        record_spawns(acp.adapter, log)
        gate = agent.exits_slowly()
        restartable = asyncio.create_task(acp.adapter.cleanup_all(final=False))
        await gate.received.wait()
        parked = asyncio.create_task(acp.send("parked", room="room-2"))
        final = asyncio.create_task(acp.adapter.cleanup_all(final=True))
        gate.release.set()
        await asyncio.gather(restartable, final)

        async with asyncio.timeout(5):
            with pytest.raises(ACPAdapterShutDownError):
                await parked
        with pytest.raises(ACPAdapterShutDownError):
            await acp.send("late", room="room-3")
        assert log == []
        assert acp.adapter._runtimes == {}
        assert acp.adapter._workspaces.rooms == ()
        await acp.adapter.on_started("Fake Agent", "restarted")


async def test_on_started_waits_for_an_overlapping_final_cleanup() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")

    async with acp_adapter(agent, fake_agent_config(inject_band_tools=True)) as acp:
        await acp.send("before", room="room-1")
        gate = agent.exits_slowly()
        shutdown = asyncio.create_task(acp.adapter.cleanup_all(final=True))
        await gate.received.wait()
        started = asyncio.create_task(acp.adapter.on_started("Fake Agent", "again"))
        gate.release.set()
        await asyncio.gather(shutdown, started)

        # The restart is not undone by the shutdown it overlapped.
        reply = await acp.send("after restart", room="room-2")

    assert reply.thoughts == ["ok"]


async def test_failed_close_keeps_the_band_server_and_retry_releases_it() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    sleeper = await spawn_sleeper()
    failing_wait_once(sleeper)

    async with acp_adapter(agent, fake_agent_config(inject_band_tools=True)) as acp:
        await acp.send("hello", room="room-1")
        await acp.send("hello", room="room-2")
        acp.adapter._runtimes["room-1"]._process = sleeper

        with pytest.raises(ACPProcessCloseError):
            await acp.adapter.cleanup_all(final=True)

        # The independent room is released; the unconfirmed one and the Band
        # server it may still dial are kept.
        assert list(acp.adapter._runtimes) == ["room-1"]
        assert acp.adapter._workspaces.rooms == ("room-1",)
        assert acp.adapter._mcp.current is not None

        await acp.adapter.cleanup_all(final=True)

        assert acp.adapter._runtimes == {}
        assert acp.adapter._mcp.current is None
        assert sleeper.returncode is not None

        await acp.adapter.on_started("Fake Agent", "restarted")
        reply = await acp.send("after restart", room="room-1")

    assert reply.thoughts == ["ok"]


class ReentrantCallback(HeldCallback):
    """A setup that asks for teardown from inside itself, then settles.

    ``cancel`` picks how it is already being cancelled when it asks: ``self``
    cancels its own task and catches it, ``shutdown`` waits until an outside
    shutdown cancels it, ``none`` asks with no cancellation outstanding.
    """

    def __init__(self, log: list[str], cancel: str) -> None:
        super().__init__(log)
        self.cancel = cancel
        self.request: Any = None

    async def __call__(self, workspace: str) -> list[ResolvedMcpServer]:
        self.calls += 1
        if self.calls > 1:
            return [http_server()]
        task = asyncio.current_task()
        assert task is not None
        if self.cancel == "self":
            task.cancel()
        if self.cancel != "none":
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancellations += 1
        else:
            self.entered.set()
        await self.request()
        self.log.append("request returned")
        await self.settlement()
        return [http_server()]


def terminate_setups(adapter: Any) -> None:
    """End setups a deadlocked teardown waits on, so a failing test can finish."""
    for initializer in list(adapter._session_initializers.values()):
        initializer.task.cancel()
    for retirement in list(adapter._retirements.values()):
        if retirement.setup is not None:
            retirement.setup.cancel()


@pytest.mark.parametrize(
    ("cancel", "request_kind", "final"),
    [
        ("self", "cleanup_all", False),
        ("self", "cleanup_all", True),
        ("none", "on_cleanup then cleanup_all", False),
        ("self", "stop", False),
        ("shutdown", "cleanup_all", True),
    ],
)
async def test_a_setup_requesting_teardown_does_not_wait_on_itself(
    cancel: str, request_kind: str, final: bool
) -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    log: list[str] = []
    callback = ReentrantCallback(log, cancel)

    async with acp_adapter(
        agent,
        fake_agent_config(inject_band_tools=True),
        resolve_mcp_servers=callback,
    ) as acp:
        adapter = acp.adapter

        async def request() -> None:
            record_close(adapter, DEFAULT_ROOM, log)
            if request_kind == "stop":
                await adapter.stop()
                return
            if request_kind.startswith("on_cleanup"):
                await adapter.on_cleanup(DEFAULT_ROOM)
            await adapter.cleanup_all(final=final)

        callback.request = request
        turn = asyncio.create_task(acp.send("hello"))
        await callback.entered.wait()
        outside = None
        if cancel == "shutdown":
            # An outside shutdown already exists when the setup asks to join it.
            outside = asyncio.create_task(adapter.cleanup_all(final=final))
        try:
            async with asyncio.timeout(5):
                await callback.settling.wait()
        except TimeoutError:
            terminate_setups(adapter)
            raise

        owned = [turn] if outside is None else [turn, outside]
        try:
            # The request returned while the setup is unsettled: everything
            # the teardown must release is still held.
            assert log == ["request returned"]
            assert DEFAULT_ROOM in adapter._runtimes
            assert DEFAULT_ROOM in adapter._workspaces.rooms
            assert adapter._mcp.current is not None
        finally:
            # Released and drained even when an assertion above failed, so
            # the fixture's teardown is never stranded behind held work.
            callback.settle.set()
            await asyncio.gather(*owned, return_exceptions=True)

        with pytest.raises(asyncio.CancelledError):
            await turn
        if outside is not None:
            await outside
        await adapter.cleanup_all(final=final)

        assert log == ["request returned", "settled", "close entered", "closed"]
        assert callback.cancellations == (0 if cancel == "none" else 1)
        assert agent.sessions == []
        assert adapter._runtimes == {}
        assert adapter._workspaces.rooms == ()
        assert adapter._mcp.current is None

        if final:
            with pytest.raises(ACPAdapterShutDownError):
                await acp.send("refused", room="room-2")
            await adapter.on_started("Fake Agent", "restarted")
        reply = await acp.send("after", room="room-2")

    assert reply.thoughts == ["ok"]


@requires_eager_tasks
async def test_a_final_request_after_shutdown_finished_runs_fresh_final_work() -> None:
    """A final request arriving once a restartable shutdown finished, before
    its registration is cleared, still stops the Band server for good: a
    stale turn continuation then neither restarts it nor opens a session."""
    agent = FakeACPAgent(http=True).will_say("ok")
    loop = asyncio.get_running_loop()
    late: list[asyncio.Task[None]] = []

    async with acp_adapter(agent, fake_agent_config(inject_band_tools=True)) as acp:
        adapter = acp.adapter
        await acp.send("before", room="room-1")
        stale = adapter._runtimes["room-1"]
        sessions = len(agent.sessions)
        backend = adapter._mcp.current
        assert backend is not None
        stop = backend.stop

        def request_final() -> None:
            # Runs after the shutdown task finished, before its done callback;
            # started eagerly so the request is made right then.
            with eager_tasks():
                late.append(asyncio.create_task(adapter.cleanup_all(final=True)))

        async def stopping() -> None:
            await stop()
            loop.call_soon(request_final)

        object.__setattr__(backend, "stop", stopping)
        await adapter.cleanup_all(final=False)
        await late[0]
        # The late final request really ran final work: the backend refuses.
        with pytest.raises(BandMCPBackendStoppedError):
            await adapter._mcp.ensure()

        with pytest.raises((asyncio.CancelledError, BandMCPBackendStoppedError)):
            await adapter._get_or_create_session(stale, "room-1", None)
        assert adapter._mcp.current is None
        assert len(agent.sessions) == sessions
        with pytest.raises(ACPAdapterShutDownError):
            await acp.send("refused", room="room-2")

        await adapter.on_started("Fake Agent", "restarted")
        reply = await acp.send("after restart", room="room-2")

    assert reply.thoughts == ["ok"]


class CountingResolver(Resolver):
    @property
    def calls(self) -> int:
        return len(self.workspaces)


async def test_a_runtime_kept_for_close_retry_serves_no_captured_turn() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")
    resolver = CountingResolver(http_server())
    sleeper = await spawn_sleeper()
    failing_wait_once(sleeper)

    async with acp_adapter(
        agent, fake_agent_config(), resolve_mcp_servers=resolver
    ) as acp:
        adapter = acp.adapter
        await acp.send("before")
        captured = adapter._runtimes[DEFAULT_ROOM]
        captured._process = sleeper
        sessions = len(agent.sessions)

        with pytest.raises(ACPProcessCloseError):
            await adapter.cleanup_all(final=False)
        assert captured.closed and captured.close_pending

        # A continuation that took the runtime before the failed cleanup.
        with pytest.raises(asyncio.CancelledError):
            await adapter._get_or_create_session(captured, DEFAULT_ROOM, None)

        assert resolver.calls == 1
        assert len(agent.sessions) == sessions
        # Still owned, for the close retry.
        assert adapter._runtimes[DEFAULT_ROOM] is captured
        assert DEFAULT_ROOM in adapter._workspaces.rooms
        assert captured._process is sleeper

        await adapter.cleanup_all(final=False)
        assert sleeper.returncode is not None
        reply = await acp.send("after")

    assert reply.thoughts == ["ok"]
    assert resolver.calls == 2


async def test_a_stale_turn_after_restartable_cleanup_starts_no_band_server() -> None:
    agent = FakeACPAgent(http=True).will_say("ok")

    async with acp_adapter(agent, fake_agent_config(inject_band_tools=True)) as acp:
        adapter = acp.adapter
        await acp.send("before")
        stale = adapter._runtimes[DEFAULT_ROOM]
        sessions = len(agent.sessions)
        await adapter.cleanup_all(final=False)
        assert adapter._mcp.current is None

        with pytest.raises(asyncio.CancelledError):
            await adapter._get_or_create_session(stale, DEFAULT_ROOM, None)

        # Not final, so nothing but the fence kept the server down.
        assert adapter._mcp.current is None
        assert len(agent.sessions) == sessions

        reply = await acp.send("after")
        assert adapter._mcp.current is not None

    assert reply.thoughts == ["ok"]
