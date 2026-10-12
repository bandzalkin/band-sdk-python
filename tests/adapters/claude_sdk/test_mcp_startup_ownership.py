"""Who owns a Claude connection while its MCP servers resolve: the room's
startup, until it publishes or settles. The CLI is the scripted fake behind
the real SDK client."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from band.integrations.claude_sdk.mcp_startup import ClaudeMcpStartupError
from band.integrations.claude_sdk.session_manager import (
    ClaudeSessionManager,
    ClaudeSessionManagerStoppedError,
)
from band.runtime.tools import BAND_MCP_SERVER_NAME
from band.workspaces import RoomWorkspaces
from tests.adapters.claude_sdk.fakecli import FakeClaude

BAND = {BAND_MCP_SERVER_NAME: {"type": "http", "url": "http://127.0.0.1:9/band"}}


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
    tmp_path: Path, resolver: Any, claude: FakeClaude
) -> ClaudeSessionManager:
    claude.mcp_statuses.append([{"name": BAND_MCP_SERVER_NAME, "status": "connected"}])
    return ClaudeSessionManager(
        ClaudeAgentOptions(),
        mcp_servers_factory=lambda room_id: BAND,
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
        with pytest.raises(RuntimeError, match="cleanup failed"):
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
