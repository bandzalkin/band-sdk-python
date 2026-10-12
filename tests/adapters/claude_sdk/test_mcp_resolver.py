"""``resolve_mcp_servers``: one frozen snapshot per new Claude connection, sent
through a private file, used only once every server reports connected."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from band.integrations.claude_sdk.mcp_startup import ClaudeMcpStartupError
from band.runtime.tools import BAND_MCP_SERVER_NAME
from tests.adapters.claude_sdk.external_mcp import external_mcp
from tests.adapters.claude_sdk.fakecli import FakeClaude, FakeCLISession
from tests.adapters.claude_sdk.helpers import ClaudeRoom
from tests.baseline.decisions import ModelDecision

OpenRoom = Callable[..., Awaitable[ClaudeRoom]]

GRANT_A = "grant-a-5ecret"
GRANT_B = "grant-b-5ecret"


class Resolver:
    """Returns ``servers`` and records each workspace it was asked for."""

    def __init__(self, servers: Mapping[str, Any]) -> None:
        self.servers = servers
        self.calls: list[str] = []

    async def __call__(self, workspace: str) -> Mapping[str, Any]:
        self.calls.append(workspace)
        return self.servers


def launch_file(session: FakeCLISession) -> Path:
    servers = session.options.mcp_servers
    assert isinstance(servers, str), "a resolver's servers go through a file"
    return Path(servers)


def connected(*names: str) -> list[dict[str, Any]]:
    return [{"name": name, "status": "connected"} for name in names]


async def test_a_new_connection_dials_its_snapshot_and_reuse_never_resolves(
    claude_room: OpenRoom, claude: FakeClaude, tmp_path: Path
) -> None:
    async with external_mcp(GRANT_A) as ext:
        resolver = Resolver({"ext": ext.server(GRANT_A)})
        room = await claude_room(
            workspace_for_room=lambda room_id: str(tmp_path / room_id),
            resolve_mcp_servers=resolver,
        )
        claude.script([room.model_reply("One.")], [room.model_reply("Two.")])
        await room.send("first")
        await room.send("second")

        assert resolver.calls == [str(tmp_path / "room-1")]
        [session] = claude.sessions
        assert set(session.configured_servers) == {"ext", BAND_MCP_SERVER_NAME}
        # A real MCP handshake with grant A happened before the first prompt.
        assert (GRANT_A, 200) in ext.receipts
        assert room.chat == ["One.", "Two."]


async def test_an_empty_result_launches_only_band(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    claude.script([room.model_reply("Hi.")])
    await room.send("hi")

    assert set(claude.sessions[0].configured_servers) == {BAND_MCP_SERVER_NAME}


async def test_the_launch_ignores_changes_to_what_the_callback_returned(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    async with external_mcp(GRANT_A) as ext:
        returned = {"ext": ext.server(GRANT_A)}

        async def resolver(workspace: str) -> Mapping[str, Any]:
            # The callback keeps changing its objects after handing them over.
            asyncio.get_running_loop().call_soon(
                returned["ext"]["headers"].__setitem__, "Authorization", "Bearer x"
            )
            return returned

        room = await claude_room(resolve_mcp_servers=resolver)
        claude.script([room.model_reply("Hi.")])
        await room.send("hi")

        sent = json.loads(launch_file(claude.sessions[0]).read_text())
        assert sent["mcpServers"]["ext"]["headers"] == {
            "Authorization": f"Bearer {GRANT_A}"
        }


async def test_a_rotated_grant_reaches_only_a_new_connection(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    async with external_mcp(GRANT_A) as ext:
        resolver = Resolver({"ext": ext.server(GRANT_A)})
        room = await claude_room(resolve_mcp_servers=resolver)
        claude.script(
            [room.model_reply("A.")],
            [room.model_reply("Still A.")],
            [room.model_reply("B.")],
        )
        await room.send("one")
        ext.grants = {GRANT_B}
        resolver.servers = {"ext": ext.server(GRANT_B)}
        await room.send("two")  # reused: no resolution, no restart
        assert len(resolver.calls) == 1
        assert len(claude.sessions) == 1

        await room.leave()
        await room.send("three")

        assert len(resolver.calls) == 2
        assert len(claude.sessions) == 2
        assert not claude.sessions[0].alive
        assert (GRANT_B, 200) in ext.receipts
        assert room.chat == ["A.", "Still A.", "B."]


async def test_a_replaced_band_server_gets_a_new_connection(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    resolver = Resolver({})
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("Before.")], [room.model_reply("After.")])
    await room.send("one")
    first = claude.sessions[0].configured_servers[BAND_MCP_SERVER_NAME]["url"]
    await room.crash_band_server()
    await room.send("two")

    assert len(claude.sessions) == 2
    assert not claude.sessions[0].alive
    second = claude.sessions[1].configured_servers[BAND_MCP_SERVER_NAME]["url"]
    assert first != second
    assert len(resolver.calls) == 2
    assert room.chat == ["Before.", "After."]


async def test_the_first_prompt_waits_for_every_server_to_connect(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    pending = [
        {"name": "ext", "status": "pending"},
        {"name": BAND_MCP_SERVER_NAME, "status": "connected"},
    ]
    claude.mcp_statuses.extend(
        [pending, pending, connected("ext", BAND_MCP_SERVER_NAME)]
    )
    resolver = Resolver({"ext": {"type": "stdio", "command": "ext-server"}})
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("Ready.")])
    await room.send("hi")

    assert claude.status_requests == 3
    assert room.chat == ["Ready."]


@pytest.mark.parametrize(
    "statuses",
    [
        pytest.param(
            [{"name": "ext", "status": "failed", "error": GRANT_A}], id="failed"
        ),
        pytest.param([{"name": "ext", "status": "needs-auth"}], id="needs-auth"),
        pytest.param([{"name": "ext", "status": "disabled"}], id="disabled"),
        pytest.param(
            [*connected("ext", BAND_MCP_SERVER_NAME), *connected("ext")], id="duplicate"
        ),
        pytest.param([{"name": "ext"}], id="malformed"),
        pytest.param([{"name": "ext", "status": "weird"}], id="unknown"),
    ],
)
async def test_a_server_that_cannot_connect_fails_the_turn_safely(
    claude_room: OpenRoom,
    claude: FakeClaude,
    caplog: pytest.LogCaptureFixture,
    statuses: list[dict[str, Any]],
) -> None:
    claude.mcp_statuses.append(
        [{"config": {"headers": {"Authorization": GRANT_A}}, **s} for s in statuses]
    )
    resolver = Resolver(
        {"ext": {"type": "stdio", "command": "ext", "env": {"T": GRANT_A}}}
    )
    room = await claude_room(resolve_mcp_servers=resolver)

    with caplog.at_level(logging.DEBUG), pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi")

    assert claude.prompts == []
    [session] = claude.sessions
    assert not session.alive
    assert not launch_file(session).exists()
    assert len(room.failures) == 1
    assert_secret_free(GRANT_A, caught.value, caplog.text)


@pytest.mark.looptime
async def test_servers_still_pending_at_the_deadline_fail_the_turn(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    resolver = Resolver({"ext": {"type": "stdio", "command": "ext"}})
    room = await claude_room(resolve_mcp_servers=resolver)
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(ClaudeMcpStartupError, match="deadline"):
        await room.send("hi")

    assert 30 <= loop.time() - started < 31
    assert claude.prompts == []
    assert not claude.sessions[0].alive


def assert_secret_free(secret: str, error: BaseException, logs: str) -> None:
    rendered = [str(error), repr(error), logs]
    chained: BaseException | None = error
    while chained is not None:
        rendered.append(str(chained))
        chained = chained.__cause__ or (
            None if chained.__suppress_context__ else chained.__context__
        )
    assert not any(secret in text for text in rendered)


async def test_a_failing_resolver_fails_once_without_falling_back(
    claude_room: OpenRoom, claude: FakeClaude, caplog: pytest.LogCaptureFixture
) -> None:
    async def resolver(workspace: str) -> Mapping[str, Any]:
        raise LookupError(f"grant store said {GRANT_A}")

    room = await claude_room(resolve_mcp_servers=resolver)
    with caplog.at_level(logging.DEBUG), pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi", session_id="sess-earlier")

    # No retry without the history, and no CLI was ever started.
    assert claude.sessions == []
    assert caught.value.stage == "resolution"
    assert caught.value.error_type == "LookupError"
    assert_secret_free(GRANT_A, caught.value, caplog.text)


@pytest.mark.parametrize(
    "servers",
    [
        pytest.param(
            {BAND_MCP_SERVER_NAME: {"type": "http", "url": "http://x"}}, id="band-name"
        ),
        pytest.param({"inproc": {"type": "sdk", "name": "inproc"}}, id="sdk"),
        pytest.param({"bad": {"type": "http"}}, id="no-url"),
    ],
)
async def test_unacceptable_servers_are_refused_by_name_before_launch(
    claude_room: OpenRoom, claude: FakeClaude, servers: dict[str, Any]
) -> None:
    room = await claude_room(resolve_mcp_servers=Resolver(servers))
    with pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi")

    assert claude.sessions == []
    assert caught.value.server == next(iter(servers))


async def test_an_external_tool_still_needs_host_permission(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    async with external_mcp(GRANT_A) as ext:
        room = await claude_room(
            resolve_mcp_servers=Resolver({"ext": ext.server(GRANT_A)})
        )
        claude.script(
            [
                ModelDecision.call("mcp__ext__echo", text="hello"),
                room.model_reply("Done."),
            ]
        )
        await room.send("hi")

        assert room.tool_errors == {"mcp__ext__echo": True, "band_send_message": False}
        assert room.chat == ["Done."]


async def test_the_native_launch_carries_a_private_file_path_not_credentials(
    claude_room: OpenRoom, claude: FakeClaude, tmp_path: Path
) -> None:
    resolver = Resolver(
        {
            "ext": {
                "type": "http",
                "url": "http://127.0.0.1:9/mcp",
                "headers": {"Authorization": f"Bearer {GRANT_A}"},
            }
        }
    )
    claude.mcp_statuses.append(connected("ext", BAND_MCP_SERVER_NAME))
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("Hi.")])
    await room.send("hi")

    [session] = claude.sessions
    path = launch_file(session)
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    native = SubprocessCLITransport(
        prompt="",
        options=ClaudeAgentOptions(
            cli_path=sys.executable, mcp_servers=session.options.mcp_servers
        ),
    )
    argv = native._build_command()
    assert argv[argv.index("--mcp-config") + 1] == str(path)
    assert not any(GRANT_A in arg for arg in argv)

    await room.leave()
    assert not path.exists()
    assert not path.parent.exists()
