"""Who owns a Claude connection while its MCP servers resolve: the room's
startup, until it publishes or settles. The CLI is the scripted fake behind
the real SDK client."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from band.adapters.claude_sdk import ClaudeSDKAdapter
from band.integrations.claude_sdk.mcp_startup import ClaudeMcpStartupError
from band.integrations.claude_sdk.session_manager import (
    ClaudeSessionManager,
    ClaudeSessionManagerStoppedError,
)
from band.runtime.tools import BAND_MCP_SERVER_NAME
from band.workspaces import RoomWorkspaces
from tests.adapters.claude_sdk.fakecli import FakeClaude, Hold

BAND = {BAND_MCP_SERVER_NAME: {"type": "http", "url": "http://127.0.0.1:9/band"}}
SECRET = "close-5ecret"


class HeldResolver:
    """Resolves only when released; records calls and cancellations. With
    ``swallow`` it finishes its own work despite a cancellation and returns."""

    def __init__(self, *, swallow: bool = False) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.swallow = swallow
        self.calls: list[str] = []
        self.cancellations = 0
        self.settled = asyncio.Event()

    async def __call__(self, workspace: str) -> Mapping[str, Any]:
        self.calls.append(workspace)
        self.entered.set()
        try:
            while True:
                try:
                    await self.release.wait()
                    break
                except asyncio.CancelledError:
                    self.cancellations += 1
                    if not self.swallow:
                        raise
            return {}
        finally:
            self.settled.set()


def manager_for(
    tmp_path: Path,
    resolver: Any,
    claude: FakeClaude,
    band: dict[str, Any] | None = None,
) -> ClaudeSessionManager:
    """A manager whose protected Band server is ``band`` (changed in place to
    model a replaced Band server)."""
    protected = BAND if band is None else band
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    return ClaudeSessionManager(
        ClaudeAgentOptions(),
        mcp_servers_factory=lambda room_id: dict(protected),
        workspaces=RoomWorkspaces(lambda room_id: str(tmp_path / room_id)),
        resolve_mcp_servers=resolver,
    )


async def test_concurrent_callers_share_one_resolution(
    tmp_path: Path, claude: FakeClaude
) -> None:
    resolver = HeldResolver()
    manager = manager_for(tmp_path, resolver, claude)
    try:
        first = asyncio.create_task(manager.get_or_create_session("room-a"))
        second = asyncio.create_task(manager.get_or_create_session("room-a"))
        await resolver.entered.wait()
        resolver.release.set()
        a, b = await asyncio.gather(first, second)
        assert a is b
        assert len(resolver.calls) == 1
        assert len(claude.sessions) == 1
    finally:
        resolver.release.set()
        await manager.stop()


async def test_a_held_resolution_never_blocks_another_room(
    tmp_path: Path, claude: FakeClaude
) -> None:
    held = HeldResolver()

    async def resolver(workspace: str) -> Mapping[str, Any]:
        if workspace.endswith("room-a"):
            return await held(workspace)
        return {}

    manager = manager_for(tmp_path, resolver, claude)
    try:
        blocked = asyncio.create_task(manager.get_or_create_session("room-a"))
        await held.entered.wait()
        other = await manager.get_or_create_session("room-b")
        await manager.cleanup_session("room-b")
        assert not blocked.done()
        assert other is not None
    finally:
        held.release.set()
        await asyncio.gather(blocked, return_exceptions=True)
        await manager.stop()


async def test_the_last_waiter_leaving_retires_the_startup_without_a_late_client(
    tmp_path: Path, claude: FakeClaude
) -> None:
    resolver = HeldResolver(swallow=True)
    manager = manager_for(tmp_path, resolver, claude)
    try:
        staying = asyncio.create_task(manager.get_or_create_session("room-a"))
        leaving = asyncio.create_task(manager.get_or_create_session("room-a"))
        await resolver.entered.wait()
        leaving.cancel()
        await asyncio.gather(leaving, return_exceptions=True)
        # One waiter still wants it: nothing was cancelled.
        assert resolver.cancellations == 0
        staying.cancel()
        await asyncio.gather(staying, return_exceptions=True)
        # The callback swallows that one cancellation and keeps going; the room
        # keeps its claim until it settles, and nothing launches afterwards.
        assert manager.owns_resources()
        resolver.release.set()
        await resolver.settled.wait()
        await manager.cleanup_session("room-a")
        assert resolver.cancellations == 1
        assert claude.sessions == []
        assert not manager.owns_resources()
    finally:
        resolver.release.set()
        await manager.stop()


async def test_a_request_cancelled_while_queued_claims_nothing(
    tmp_path: Path, claude: FakeClaude
) -> None:
    resolver = HeldResolver()
    manager = manager_for(tmp_path, resolver, claude)
    await manager.start()
    try:
        request = asyncio.create_task(manager.get_or_create_session("room-a"))
        await asyncio.sleep(0)  # queued, not yet reserved
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        # A later command proves the queue already passed the cancelled one.
        await manager.cleanup_session("room-other")
        assert resolver.calls == []
        assert not manager.owns_resources()
    finally:
        await manager.stop()


@pytest.mark.parametrize("swallow", [False, True])
async def test_cleanup_retires_a_resolving_startup_and_holds_the_room_until_it_settles(
    tmp_path: Path, claude: FakeClaude, swallow: bool
) -> None:
    resolver = HeldResolver(swallow=swallow)
    manager = manager_for(tmp_path, resolver, claude)
    try:
        waiting = asyncio.create_task(manager.get_or_create_session("room-a"))
        await resolver.entered.wait()
        cleanup = asyncio.create_task(manager.cleanup_session("room-a"))
        arrival = asyncio.create_task(manager.get_or_create_session("room-a"))
        await asyncio.wait((cleanup,), timeout=0)
        if swallow:
            assert not cleanup.done()
            resolver.release.set()
        await cleanup
        with pytest.raises((ClaudeMcpStartupError, asyncio.CancelledError)):
            await waiting
        # The startup's late result launched nothing; the arrival during the
        # retirement got a fresh startup only after it.
        assert resolver.cancellations == 1
        resolver.release.set()
        client = await arrival
        assert len(resolver.calls) == 2
        assert len(claude.sessions) == 1
        assert claude.sessions[0].alive and client is not None
    finally:
        resolver.release.set()
        await manager.stop()


async def test_a_resolver_cleaning_up_its_own_room_does_not_wait_on_itself(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager_ref: list[ClaudeSessionManager] = []
    returned = asyncio.Event()

    async def resolver(workspace: str) -> Mapping[str, Any]:
        await manager_ref[0].cleanup_session("room-a")
        returned.set()
        return {}

    manager = manager_for(tmp_path, resolver, claude)
    manager_ref.append(manager)
    try:
        with pytest.raises(ClaudeMcpStartupError):
            async with asyncio.timeout(5):
                await manager.get_or_create_session("room-a")
        assert returned.is_set()
        assert claude.sessions == []
        async with asyncio.timeout(5):
            while manager.owns_resources():
                await asyncio.sleep(0)
    finally:
        await manager.stop()


async def test_a_resolver_stopping_the_manager_does_not_wait_on_itself(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager_ref: list[ClaudeSessionManager] = []

    async def resolver(workspace: str) -> Mapping[str, Any]:
        await manager_ref[0].stop()
        return {}

    manager = manager_for(tmp_path, resolver, claude)
    manager_ref.append(manager)
    with pytest.raises(ClaudeSessionManagerStoppedError):
        async with asyncio.timeout(5):
            await manager.get_or_create_session("room-a")
    async with asyncio.timeout(5):
        await manager.stop()
    assert claude.sessions == []
    assert not manager.owns_resources()


async def test_stop_wins_over_a_startup_finishing_late(
    tmp_path: Path, claude: FakeClaude
) -> None:
    resolver = HeldResolver(swallow=True)
    manager = manager_for(tmp_path, resolver, claude)
    waiting = asyncio.create_task(manager.get_or_create_session("room-a"))
    await resolver.entered.wait()
    stopping = asyncio.create_task(manager.stop())
    await asyncio.wait((stopping,), timeout=0)
    resolver.release.set()
    await stopping
    with pytest.raises(ClaudeSessionManagerStoppedError):
        await waiting
    assert claude.sessions == []
    assert not manager.owns_resources()


async def test_a_failed_close_keeps_the_file_and_claim_until_a_retry(
    tmp_path: Path, claude: FakeClaude
) -> None:
    async def resolver(workspace: str) -> Mapping[str, Any]:
        return {}

    manager = manager_for(tmp_path, resolver, claude)
    try:
        await manager.get_or_create_session("room-a")
        [session] = claude.sessions
        path = Path(str(session.options.mcp_servers))
        claude.refuse_close = True
        with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
            await manager.cleanup_session("room-a")
        assert path.exists()
        assert manager.owns_resources()

        claude.refuse_close = False
        await manager.cleanup_session("room-a")
        assert not path.exists()
        assert not manager.owns_resources()
    finally:
        claude.refuse_close = False
        await manager.stop()


class SettlingResolver:
    """Already being cancelled (by itself or its own room cleanup) when it
    enters a settlement it must finish; logs any further cancellation."""

    def __init__(self, *, retire_own_room: bool = False) -> None:
        self.retire_own_room = retire_own_room
        self.manager: ClaudeSessionManager | None = None
        self.settling = asyncio.Event()
        self.release = asyncio.Event()
        self.log: list[str] = []

    async def __call__(self, workspace: str) -> Mapping[str, Any]:
        if self.retire_own_room:
            assert self.manager is not None
            await self.manager.cleanup_session("room-a")
        else:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass  # caught, never uncancelled
        self.settling.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.log.append("settlement interrupted")
            raise
        self.log.append("settled")
        return {}


@pytest.mark.parametrize(
    "retire_own_room", [False, True], ids=["self-cancel", "self-cleanup"]
)
async def test_a_settling_startup_is_never_cancelled_again(
    tmp_path: Path, claude: FakeClaude, retire_own_room: bool
) -> None:
    resolver = SettlingResolver(retire_own_room=retire_own_room)
    manager = manager_for(tmp_path, resolver, claude)
    resolver.manager = manager
    try:
        waiting = asyncio.create_task(manager.get_or_create_session("room-a"))
        await resolver.settling.wait()
        # An outside cleanup takes the startup over while it settles.
        cleanup = asyncio.create_task(manager.cleanup_session("room-a"))
        await asyncio.sleep(0)
        resolver.release.set()
        await cleanup
        with pytest.raises((ClaudeMcpStartupError, asyncio.CancelledError)):
            await waiting
        assert resolver.log == ["settled"]
        assert claude.sessions == []
        assert not manager.owns_resources()
    finally:
        resolver.release.set()
        await manager.stop()


class CountingResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, workspace: str) -> Mapping[str, Any]:
        self.calls.append(workspace)
        return {}


async def outdated_session(
    tmp_path: Path, claude: FakeClaude
) -> tuple[ClaudeSessionManager, CountingResolver, dict[str, Any]]:
    """A manager whose room-a session (A) dials a Band server since replaced."""
    resolver = CountingResolver()
    band = dict(BAND)
    manager = manager_for(tmp_path, resolver, claude, band)
    await manager.get_or_create_session("room-a")
    band[BAND_MCP_SERVER_NAME] = {"type": "http", "url": "http://127.0.0.1:9/band-2"}
    return manager, resolver, band


async def test_a_replacement_whose_caller_left_during_the_old_close_never_publishes(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager, resolver, _ = await outdated_session(tmp_path, claude)
    hold = claude.closing = Hold()
    try:
        replacing = asyncio.create_task(manager.get_or_create_session("room-a"))
        await hold.reached.wait()  # the old client's close is in progress
        replacing.cancel()
        await asyncio.gather(replacing, return_exceptions=True)
        hold.released.set()
        await manager.cleanup_session("room-a")

        # The replacement was retired: never resolved, launched or published.
        assert len(resolver.calls) == 1
        assert len(claude.sessions) == 1
        assert not claude.sessions[0].alive
        assert not manager.owns_resources()
    finally:
        hold.released.set()
        claude.closing = None
        await manager.stop()


async def test_stop_during_a_replacements_old_close_leaves_nothing_behind(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager, resolver, _ = await outdated_session(tmp_path, claude)
    hold = claude.closing = Hold()
    try:
        replacing = asyncio.create_task(manager.get_or_create_session("room-a"))
        await hold.reached.wait()
        stopping = asyncio.create_task(manager.stop())
        hold.released.set()
        await stopping
        with pytest.raises(ClaudeSessionManagerStoppedError):
            await replacing
        await manager.stop()

        assert len(resolver.calls) == 1
        assert len(claude.sessions) == 1
        assert not manager.owns_resources()
    finally:
        hold.released.set()
        claude.closing = None


async def test_a_resolver_that_swallows_its_own_cancellation_publishes_nothing(
    tmp_path: Path, claude: FakeClaude
) -> None:
    async def resolver(workspace: str) -> Mapping[str, Any]:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass  # caught, never uncancelled, and a result returned anyway
        return {}

    manager = manager_for(tmp_path, resolver, claude)
    try:
        with pytest.raises(ClaudeMcpStartupError):
            await manager.get_or_create_session("room-a")
        assert claude.sessions == []
        assert not manager.has_session("room-a")
    finally:
        await manager.stop()


async def test_a_finished_keep_claim_retirement_does_not_stall_a_releasing_cleanup(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager = manager_for(tmp_path, CountingResolver(), claude)
    await manager.get_or_create_session("room-a")
    try:
        # A membership-keeping retirement that finished, its removal callback
        # not yet run.
        keeping = manager._begin_room_retirement("room-a", release_workspace=False)
        await asyncio.shield(keeping.task)
        manager._retirements["room-a"] = keeping

        await manager.cleanup_session("room-a")

        assert not manager.owns_resources()
    finally:
        await manager.stop()


async def test_a_file_that_cannot_be_removed_stays_owned_until_a_retry(
    tmp_path: Path, claude: FakeClaude
) -> None:
    manager = manager_for(tmp_path, CountingResolver(), claude)
    await manager.get_or_create_session("room-a")
    path = Path(str(claude.sessions[0].options.mcp_servers))
    path.parent.chmod(0o500)
    try:
        with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
            await manager.cleanup_session("room-a")
        assert path.exists()
        assert manager.owns_resources()

        path.parent.chmod(0o700)
        await manager.cleanup_session("room-a")
        assert not path.exists()
        assert not manager.owns_resources()
    finally:
        if path.parent.exists():
            path.parent.chmod(0o700)
        await manager.stop()


async def test_a_close_failure_reports_no_native_text(
    tmp_path: Path, claude: FakeClaude, caplog: pytest.LogCaptureFixture
) -> None:
    manager = manager_for(tmp_path, CountingResolver(), claude)
    await manager.get_or_create_session("room-a")
    [session] = claude.sessions
    real_close = session.close

    async def leaking_close() -> None:
        raise OSError(f"close failed with {SECRET}")

    session.close = leaking_close  # type: ignore[method-assign]
    try:
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(ClaudeMcpStartupError) as caught,
        ):
            await manager.cleanup_session("room-a")
        rendered = [str(caught.value), repr(caught.value), caplog.text]
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        assert not any(SECRET in text for text in rendered)
        assert manager.owns_resources()
    finally:
        session.close = real_close  # type: ignore[method-assign]
        await manager.stop()


class FailingBand:
    """A protected-server factory that can be made to fail, with a secret."""

    def __init__(self) -> None:
        self.failing = False

    def __call__(self, room_id: str) -> dict[str, Any]:
        if self.failing:
            raise LookupError(f"band backend said {SECRET}")
        return dict(BAND)


@pytest.mark.parametrize("retained", ["file", "child"])
async def test_a_failed_reservation_keeps_what_the_room_still_holds(
    tmp_path: Path,
    claude: FakeClaude,
    caplog: pytest.LogCaptureFixture,
    retained: str,
) -> None:
    resolver = CountingResolver()
    band = FailingBand()
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    manager = ClaudeSessionManager(
        ClaudeAgentOptions(),
        mcp_servers_factory=band,
        workspaces=RoomWorkspaces(lambda room_id: str(tmp_path / room_id)),
        resolve_mcp_servers=resolver,
    )
    await manager.get_or_create_session("room-a")
    path = Path(str(claude.sessions[0].options.mcp_servers))
    try:
        # The room's cleanup fails, leaving its file (or its whole child) held.
        if retained == "file":
            path.parent.chmod(0o500)
        else:
            claude.refuse_close = True
        with pytest.raises(ClaudeMcpStartupError, match="cleanup"):
            await manager.cleanup_session("room-a", release_workspace=False)
        if retained == "file":
            path.parent.chmod(0o700)
        else:
            claude.refuse_close = False

        band.failing = True
        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(ClaudeMcpStartupError) as caught,
        ):
            await manager.get_or_create_session("room-a")
        assert caught.value.stage == "reservation"
        assert SECRET not in str(caught.value) and SECRET not in caplog.text
        # Nothing new was resolved, allocated or spawned, and the old custody
        # is still owned.
        assert len(resolver.calls) == 1
        assert len(claude.sessions) == 1
        assert path.exists()
        assert manager.owns_resources()

        band.failing = False
        await manager.cleanup_session("room-a")
        assert not path.exists()
        assert not manager.owns_resources()
        await manager.get_or_create_session("room-a")
        assert len(claude.sessions) == 2
    finally:
        if path.parent.exists():
            path.parent.chmod(0o700)
        claude.refuse_close = False
        band.failing = False
        await manager.stop()
    assert not manager.owns_resources()


@pytest.mark.parametrize(
    "phase", ["reservation", "reuse", "pre-launch", "construction", "post-status"]
)
async def test_a_callback_closing_admission_stops_everything_after_it(
    tmp_path: Path,
    claude: FakeClaude,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """A callback the manager runs closes its admission mid-startup: nothing
    is constructed, reused or published after that."""
    owned_tmp = tmp_path / "tmp"
    owned_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(owned_tmp))
    armed = {"on": phase == "reservation"}
    manager_ref: list[ClaudeSessionManager] = []
    resolutions: list[str] = []

    def band(room_id: str) -> dict[str, Any]:
        if armed["on"] or (phase == "post-status" and claude.status_requests > 0):
            manager_ref[0].close_admission()
        return dict(BAND)

    async def resolver(workspace: str) -> Mapping[str, Any]:
        resolutions.append(workspace)
        if phase == "pre-launch":
            armed["on"] = True
        return {}

    async def allow(*args: Any) -> Any:
        return None

    def permissions(room_id: str) -> Any:
        if phase == "construction" and resolutions:
            manager_ref[0].close_admission()
        return allow

    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    manager = ClaudeSessionManager(
        ClaudeAgentOptions(),
        can_use_tool_factory=permissions,
        mcp_servers_factory=band,
        workspaces=RoomWorkspaces(lambda room_id: str(tmp_path / room_id)),
        resolve_mcp_servers=resolver,
    )
    manager_ref.append(manager)
    try:
        if phase == "reuse":
            await manager.get_or_create_session("room-a")
            armed["on"] = True
        sessions_before = len(claude.sessions)
        with pytest.raises(ClaudeSessionManagerStoppedError):
            async with asyncio.timeout(10):
                await manager.get_or_create_session("room-a")

        if phase == "post-status":
            # Built and confirmed, then refused instead of published.
            assert len(claude.sessions) == 1
            assert not claude.sessions[0].alive
        else:
            assert len(claude.sessions) == sessions_before
        if phase == "reservation":
            assert resolutions == []
        assert not manager.has_session("room-a") or phase == "reuse"
    finally:
        await manager.stop()
    assert not manager.owns_resources()
    assert all(not s.alive for s in claude.sessions)
    assert list(owned_tmp.glob("band-claude-mcp-*")) == []


class ReuseProbe:
    """The servers a held client dials, as a callback would hand them over: the
    comparison answers with this callback's own object, whose truth is therefore
    consulted wherever the decision is read (an ``and`` chain coerces each
    earlier operand and leaves the final one, and the settled result is read
    again at the branch). Closing admission from the second read models a
    callback acting when the decision is read after the final admission guard;
    closing on the first read is the early control. The client itself stays the
    manager's real one, so its teardown stays owned."""

    def __init__(self, manager: ClaudeSessionManager, mode: str) -> None:
        self.manager = manager
        self.mode = mode
        self.reads = 0

    def _decide(self) -> bool:
        self.reads += 1
        closing_now = (
            self.reads == 1 if self.mode == "materialized" else self.reads >= 2
        )
        if closing_now:
            self.manager.close_admission()
        return True

    def __eq__(self, other: object) -> Any:
        return self

    def __bool__(self) -> bool:
        return self._decide()

    __hash__ = None  # type: ignore[assignment]


class HeldServers:
    """A stand-in for the held client's options: only its servers are read."""

    def __init__(self, servers: Any) -> None:
        self.mcp_servers = servers


class ClientCustody:
    """What teardown must leave behind on an owned client: one disconnect, its
    Query closed on both sides, and every reference dropped. A manager that
    reports nothing held while the client it owned stays connected is the
    failure this pins, so the checks read the client, not the manager."""

    def __init__(self, client: ClaudeSDKClient) -> None:
        self.client = client
        self.query = client._query
        self.disconnects = 0
        self.calls: list[str] = []
        assert self.query is not None, "the client was never connected"
        query, real_disconnect = self.query, client.disconnect
        real_close, real_close_receive = query.close, query.close_receive_stream

        async def disconnect() -> None:
            self.disconnects += 1
            await real_disconnect()

        async def close() -> None:
            self.calls.append("close")
            await real_close()

        def close_receive() -> None:
            self.calls.append("receive")
            real_close_receive()

        client.disconnect = disconnect  # type: ignore[method-assign]
        query.close = close  # type: ignore[method-assign]
        query.close_receive_stream = close_receive  # type: ignore[method-assign]

    def assert_disconnected(self) -> None:
        assert self.disconnects == 1, f"disconnect ran {self.disconnects} times"
        assert self.calls == ["close", "receive"], self.calls
        assert self.query._closed, "the client's Query stayed open"
        assert self.client._query is None
        assert self.client._transport is None
        assert self.client._materialized is None


@pytest.mark.parametrize("mode", ["materialized", "deferred"])
async def test_the_reuse_decision_is_a_plain_bool_by_the_final_guard(
    tmp_path: Path,
    claude: FakeClaude,
    mode: str,
) -> None:
    """The reuse decision is reduced to a plain bool once, inside the fallible
    prerequisite boundary, before the callback-free admission guard. An object
    answering the currentness comparison is therefore never asked its truth
    again: a callback that would close admission at the branch test cannot run
    there at all (the deferred case), and closing on the first read is refused
    outright (the control)."""

    def band(room_id: str) -> Any:
        return dict(BAND)

    manager = ClaudeSessionManager(
        ClaudeAgentOptions(),
        mcp_servers_factory=band,
        workspaces=RoomWorkspaces(lambda room_id: str(tmp_path / room_id)),
    )
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    client = await manager.get_or_create_session("room-a")
    custody = ClientCustody(client)
    real_options = client.options
    probe = ReuseProbe(manager, mode)
    sessions_before = len(claude.sessions)
    try:
        client.options = HeldServers(probe)
        if mode == "materialized":  # closes while the decision is materialized
            with pytest.raises(ClaudeSessionManagerStoppedError):
                await manager.get_or_create_session("room-a")
        else:  # a second read, which the settled decision must never take
            returned = await manager.get_or_create_session("room-a")
            assert probe.reads == 1, f"the decision was read {probe.reads} times"
            assert not manager._closing
            assert returned is client
    finally:
        client.options = real_options
        await manager.stop()
    assert all(not s.alive for s in claude.sessions)
    assert len(claude.sessions) == sessions_before
    assert not manager.owns_resources()
    custody.assert_disconnected()


class RaisingTruth:
    """A callback's servers whose comparison answers with the callback's own
    object and whose truth test raises a secret-bearing error of a builtin
    type, as a callback-computed step could."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.reads = 0

    def __eq__(self, other: object) -> Any:
        return self

    def __bool__(self) -> bool:
        self.reads += 1
        raise LookupError(self.text)

    __hash__ = None  # type: ignore[assignment]


@pytest.mark.parametrize("with_resolver", [True, False])
async def test_a_failing_reuse_decision_stays_a_safe_non_resume_failure(
    tmp_path: Path,
    claude: FakeClaude,
    caplog: pytest.LogCaptureFixture,
    with_resolver: bool,
) -> None:
    """A truth test that raises while the reuse decision is worked out is a
    reservation failure the owner built, with or without a resolver: its text
    carries no callback value, and the adapter never retries the room without
    its history."""
    secret = "res-ul3t-9a7c"

    def band(room_id: str) -> Any:
        return dict(BAND)

    async def resolver(workspace: str) -> Mapping[str, Any]:
        return {}

    manager = ClaudeSessionManager(
        ClaudeAgentOptions(),
        mcp_servers_factory=band,
        workspaces=RoomWorkspaces(lambda room_id: str(tmp_path / room_id)),
        resolve_mcp_servers=resolver if with_resolver else None,
    )
    calls: list[str | None] = []
    original = manager.get_or_create_session

    async def counted(
        room_id: str, *, resume_session_id: str | None = None
    ) -> ClaudeSDKClient:
        calls.append(resume_session_id)
        return await original(room_id, resume_session_id=resume_session_id)

    manager.get_or_create_session = counted  # type: ignore[method-assign]
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    client = await manager.get_or_create_session("room-a")
    custody = ClientCustody(client)
    probe = RaisingTruth(secret)
    real_options = client.options
    real_identity = manager._identities.get("room-a")
    calls.clear()  # only the adapter's own attempt counts
    try:
        if with_resolver:  # the retained protected servers are the comparison
            manager._identities["room-a"] = probe
        else:  # what the held client dials is the comparison
            client.options = HeldServers(probe)
        # The decision's truth test runs inside the fallible boundary; the
        # branch test after the callback-free guard reads the settled bool.
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(ClaudeMcpStartupError) as caught,
        ):
            await ClaudeSDKAdapter._open_session(manager, "room-a", "hist-1")
    finally:
        client.options = real_options
        if real_identity is not None:
            manager._identities["room-a"] = real_identity
        await manager.stop()
    assert probe.reads == 1, f"the decision was read {probe.reads} times"
    error = caught.value
    assert error.stage == "reservation" and not error.resume_eligible
    assert secret not in str(error)
    assert secret not in caplog.text
    assert calls == ["hist-1"], "the room's session was started again without history"
    assert all(not s.alive for s in claude.sessions)
    assert not manager.owns_resources()
    custody.assert_disconnected()
