"""``resolve_mcp_servers``: one frozen snapshot per new Claude connection, sent
through a private file, used only once every server reports connected."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import typing
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from claude_agent_sdk.types import McpServerConfig

from band.adapters.claude_sdk import ClaudeSDKAdapter
from band.integrations.claude_sdk.mcp_startup import (
    ClaudeMcpStartupError,
    McpConfigFile,
    startup_error,
)
from band.integrations.claude_sdk.session_manager import (
    ClaudeSessionManager,
    ClaudeSessionManagerStoppedError,
)
from band.runtime.tools import BAND_MCP_SERVER_NAME
from tests.adapters.claude_sdk.external_mcp import external_mcp
from tests.adapters.claude_sdk.fakecli import FakeClaude, FakeCLISession, Hold
from tests.adapters.claude_sdk.helpers import ClaudeRoom, with_approvals
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


async def test_a_missing_history_still_falls_back_to_a_fresh_session(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    claude.unresumable = {"sess-gone"}
    resolver = Resolver({})
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("Fresh.")])
    await room.send("hi", session_id="sess-gone")

    assert claude.resumed == ["sess-gone", None]
    # Each genuine connection resolved its own snapshot.
    assert len(resolver.calls) == 2
    assert room.chat == ["Fresh."]


class ExplodingServers(Mapping[str, Any]):
    """A mapping that fails while being read, with a credential in its error."""

    def __getitem__(self, key: str) -> Any:
        raise LookupError(f"vault said {GRANT_A}")

    def __iter__(self) -> Any:
        raise LookupError(f"vault said {GRANT_A}")

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize("failure", ["unreadable-result", "typed-error"])
async def test_callback_failures_are_sanitized_and_never_retried_fresh(
    claude_room: OpenRoom,
    claude: FakeClaude,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    calls: list[str] = []

    async def resolver(workspace: str) -> Mapping[str, Any]:
        calls.append(workspace)
        if failure == "typed-error":
            raise ClaudeMcpStartupError(
                "resolution", f"token {GRANT_A}", server=GRANT_A
            )
        return ExplodingServers()

    room = await claude_room(resolve_mcp_servers=resolver)
    with caplog.at_level(logging.DEBUG), pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi", session_id="sess-earlier")

    assert len(calls) == 1
    assert claude.sessions == []
    assert_secret_free(GRANT_A, caught.value, caplog.text)


async def test_no_prompt_before_a_held_status_reports_connected(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    hold = claude.status_hold = Hold()
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    claude.script([room.model_reply("Ready.")])
    sending = room.send_in_background("hi")
    try:
        await hold.reached.wait()
        assert claude.prompts == []
        assert not room.adapter._session_manager.has_session(room.room_id)
    finally:
        hold.released.set()
    await sending
    assert room.chat == ["Ready."]


@pytest.mark.looptime
async def test_a_status_request_that_never_answers_fails_at_the_readiness_deadline(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    hold = claude.status_hold = Hold()
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with pytest.raises(ClaudeMcpStartupError, match="deadline"):
            await room.send("hi")
        # Bounded by the 30 s total, not the SDK's own 60 s per request.
        assert 30 <= loop.time() - started < 31
        assert claude.prompts == []
    finally:
        hold.released.set()


async def test_a_cli_without_mcp_status_fails_safely(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    claude.status_unsupported = True
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    with pytest.raises(ClaudeMcpStartupError, match="status request failed"):
        await room.send("hi")
    assert claude.prompts == []
    assert not claude.sessions[0].alive


@pytest.mark.parametrize("held", ["resolution", "readiness"])
async def test_a_band_server_replaced_during_startup_gets_a_new_connection(
    claude_room: OpenRoom, claude: FakeClaude, held: str
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls: list[str] = []

    async def resolver(workspace: str) -> Mapping[str, Any]:
        calls.append(workspace)
        if held == "resolution" and len(calls) == 1:
            entered.set()
            await release.wait()
        return {}

    if held == "readiness":
        hold = claude.status_hold = Hold()
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("Live.")])
    sending = room.send_in_background("hi")
    try:
        if held == "resolution":
            await entered.wait()
        else:
            await hold.reached.wait()
        old_url = room.adapter._room_mcp_servers(room.room_id)[BAND_MCP_SERVER_NAME][
            "url"
        ]
        await room.crash_band_server()
        await room.adapter._mcp.ensure()
    finally:
        release.set()
        if held == "readiness":
            claude.status_hold = None
            hold.released.set()
    await sending

    live = claude.sessions[-1]
    assert live.alive
    assert live.configured_servers[BAND_MCP_SERVER_NAME]["url"] != old_url
    assert all(not s.alive for s in claude.sessions[:-1])
    assert len(calls) == 2
    assert room.chat == ["Live."]


async def test_a_resolver_requesting_teardown_behind_a_settling_turn_publishes_nothing(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    adapter_ref: list[Any] = []
    requested = asyncio.Event()

    async def resolver(workspace: str) -> Mapping[str, Any]:
        if workspace.endswith("room-1"):
            return {}
        await adapter_ref[0].cleanup_all()
        requested.set()
        return {}

    room = await claude_room(resolve_mcp_servers=resolver)
    adapter_ref.append(room.adapter)
    other = room.beside("room-2")
    turn_hold = Hold()
    claude.script([turn_hold])
    first = room.send_in_background("hold on")
    try:
        await turn_hold.reached.wait()
        sessions_before = len(claude.sessions)

        with pytest.raises((ClaudeMcpStartupError, ClaudeSessionManagerStoppedError)):
            await other.send("hi")
        assert requested.is_set()
    finally:
        # Released and drained whatever the assertions above found.
        turn_hold.released.set()
        await asyncio.gather(first, return_exceptions=True)
    await room.adapter.cleanup_all()

    # The room-2 startup asked for the teardown and was refused on its own:
    # nothing launched after the request.
    assert len(claude.sessions) == sessions_before
    assert not room.adapter._session_manager.owns_resources()


async def test_a_failed_teardown_keeps_turns_out_until_a_restart(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    claude.script([room.model_reply("Hi.")])
    await room.send("hi")
    backend = room.adapter._mcp.current
    bound = room.adapter._mcp_room_tools.get(room.room_id)
    claude.refuse_close = True
    try:
        with pytest.raises(BaseException, match="cleanup"):
            await room.adapter.cleanup_all()
        # The retained Band server dies; a late turn must not start another.
        await room.crash_band_server()
        with pytest.raises(ClaudeSessionManagerStoppedError):
            await room.send("again")
        assert room.adapter._mcp.current is backend
        assert room.adapter._mcp_room_tools.get(room.room_id) is bound
    finally:
        claude.refuse_close = False
    await room.adapter.on_started("Test Agent", "restarted")
    claude.script([room.model_reply("Back.")])
    await room.send("later")
    assert room.chat == ["Hi.", "Back."]


def private_files() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob("band-claude-mcp-*/mcp.json"))


@contextlib.contextmanager
def unremovable(path: Path) -> Iterator[None]:
    """Its private directory refuses removal until the block exits."""
    path.parent.chmod(0o500)
    try:
        yield
    finally:
        if path.parent.exists():
            path.parent.chmod(0o700)


async def test_a_file_left_behind_fences_new_connections_until_removed(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    before = private_files()
    resolver = Resolver({})
    room = await claude_room(resolve_mcp_servers=resolver)
    claude.script([room.model_reply("A.")], [room.model_reply("B.")])
    await room.send("first")
    first = launch_file(claude.sessions[0])
    with unremovable(first):
        with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
            await room.leave()
        # A new turn takes the left-behind file over and can't remove it either:
        # it launches, resolves and allocates nothing.
        with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
            await room.send("second")
        assert len(claude.sessions) == 1
        assert len(resolver.calls) == 1
        assert first.exists()
        assert room.adapter._session_manager.owns_resources()

    await room.send("third")
    assert not first.exists()
    assert room.chat == ["A.", "B."]
    await room.leave()
    assert private_files() == before
    assert not room.adapter._session_manager.owns_resources()


async def test_a_resume_fallback_does_not_strand_the_failed_attempts_file(
    claude_room: OpenRoom, claude: FakeClaude
) -> None:
    before = private_files()
    claude.unresumable = {"sess-gone"}
    resolver = Resolver({})
    room = await claude_room(resolve_mcp_servers=resolver)
    hold = claude.connecting = Hold()
    sending = asyncio.create_task(room.send("hi", session_id="sess-gone"))
    try:
        await hold.reached.wait()
        attempt = launch_file(claude.sessions[0])
        with unremovable(attempt):
            claude.connecting = None
            hold.released.set()
            # The resumed attempt fails; the fresh fallback finds its file
            # still held and stops there.
            with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
                await sending
            assert claude.resumed == ["sess-gone"]
            assert attempt.exists()
            assert room.adapter._session_manager.owns_resources()
    finally:
        claude.connecting = None
        hold.released.set()
        await asyncio.gather(sending, return_exceptions=True)

    claude.script([room.model_reply("Fresh.")])
    await room.send("again")
    assert not attempt.exists()
    assert room.chat == ["Fresh."]
    await room.leave()
    assert private_files() == before


async def test_a_partly_written_file_stays_owned_until_removed(
    claude_room: OpenRoom,
    claude: FakeClaude,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    before = private_files()
    written: list[Path] = []
    real_write = McpConfigFile.write

    def interrupted(self: McpConfigFile, servers: Mapping[str, Any]) -> None:
        self.path.write_text(f'{{"partial": "{GRANT_A}"')
        written.append(self.path)
        self.directory.chmod(0o500)  # and its removal is refused
        raise OSError(f"disk full writing {GRANT_A}")

    monkeypatch.setattr(McpConfigFile, "write", interrupted)
    room = await claude_room(resolve_mcp_servers=Resolver({}))
    try:
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(ClaudeMcpStartupError) as caught,
        ):
            await room.send("hi")
        assert caught.value.stage == "launch"
        assert claude.sessions == []
        assert_secret_free(GRANT_A, caught.value, caplog.text)
        [partial] = written
        assert partial.exists()
        assert room.adapter._session_manager.owns_resources()
    finally:
        for path in written:
            if path.parent.exists():
                path.parent.chmod(0o700)
    monkeypatch.setattr(McpConfigFile, "write", real_write)

    await room.leave()
    assert not written[0].exists()
    assert private_files() == before
    assert not room.adapter._session_manager.owns_resources()


async def test_a_failing_permission_factory_is_sanitized_and_never_retried_fresh(
    claude_room: OpenRoom,
    claude: FakeClaude,
    caplog: pytest.LogCaptureFixture,
) -> None:
    before = private_files()
    resolver = Resolver({})
    room = await claude_room(with_approvals(), resolve_mcp_servers=resolver)

    def leaking_factory(room_id: str) -> Any:
        raise LookupError(f"policy store said {GRANT_A}")

    room.adapter._session_manager._can_use_tool_factory = leaking_factory
    with caplog.at_level(logging.DEBUG), pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi", session_id="sess-earlier")

    assert caught.value.stage == "launch"
    assert not caught.value.resume_eligible
    assert len(resolver.calls) == 1
    assert claude.sessions == []
    assert_secret_free(GRANT_A, caught.value, caplog.text)
    assert private_files() == before


class MarkedServers(Mapping[str, Any]):
    """A lazy mapping raising the module's own safe-looking error type, with a
    credential in its reason and resume eligibility set."""

    def __getitem__(self, key: str) -> Any:
        raise startup_error("launch", f"token {GRANT_A}", resume_eligible=True)

    def __iter__(self) -> Any:
        raise startup_error("launch", f"token {GRANT_A}", resume_eligible=True)

    def __len__(self) -> int:
        return 1


async def test_an_owned_looking_error_from_the_result_is_still_untrusted(
    claude_room: OpenRoom, claude: FakeClaude, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    async def resolver(workspace: str) -> Mapping[str, Any]:
        calls.append(workspace)
        return MarkedServers()

    room = await claude_room(resolve_mcp_servers=resolver)
    with caplog.at_level(logging.DEBUG), pytest.raises(ClaudeMcpStartupError) as caught:
        await room.send("hi", session_id="sess-earlier")

    assert caught.value.stage == "resolution"
    assert len(calls) == 1
    assert claude.sessions == []
    assert_secret_free(GRANT_A, caught.value, caplog.text)


def test_the_public_constructors_have_evaluable_native_resolver_types() -> None:
    for constructor in (ClaudeSDKAdapter.__init__, ClaudeSessionManager.__init__):
        hint = typing.get_type_hints(constructor)["resolve_mcp_servers"]
        resolver, none = typing.get_args(hint)
        assert none is type(None)
        [returned] = typing.get_args(typing.get_args(resolver)[1])
        assert typing.get_args(returned) == (str, McpServerConfig)


def test_the_resolver_type_needs_no_claude_sdk_at_import() -> None:
    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'claude_agent_sdk':\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from band.integrations.claude_sdk.mcp_startup import McpServerResolver\n"
        "McpServerResolver | None\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
