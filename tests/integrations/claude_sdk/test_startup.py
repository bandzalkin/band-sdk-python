"""Retained transport ownership closes failures before SDK Query construction."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from band.integrations.claude_sdk import transport
from band.integrations.claude_sdk.session_manager import ClaudeSessionManager
from band.workspaces import RoomWorkspaces
from tests.adapters.claude_sdk.process import WorkspaceProcess


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_pre_query_startup_failure_reaps_real_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    children: list[WorkspaceProcess] = []

    def launch(*, prompt: str, options: ClaudeAgentOptions) -> WorkspaceProcess:
        child = WorkspaceProcess(prompt=prompt, options=options)
        child.refuse_close = cleanup_fails
        children.append(child)
        return child

    monkeypatch.setattr(transport, "SubprocessCLITransport", launch)
    monkeypatch.setenv("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "invalid")
    manager = ClaudeSessionManager(
        ClaudeAgentOptions(cli_path=sys.executable, env={"PROBE_MARKER": "startup"}),
        workspaces=RoomWorkspaces(lambda _: str(tmp_path / "shared")),
    )
    try:
        # A failed cleanup is retained for retry; the startup's own error stays
        # the caller's.
        with pytest.raises(ValueError):
            await manager.get_or_create_session("room-a")
        assert len(children) == 1
        if cleanup_fails:
            assert not children[0].exited
            assert manager.has_session("room-a")
            with pytest.raises(ValueError, match="both"):
                await manager.get_or_create_session("room-b")
            with pytest.raises(RuntimeError, match="cleanup failed"):
                await manager.get_or_create_session("room-a")
            assert len(children) == 1
            children[0].refuse_close = False
            await manager.cleanup_session("room-a")
        assert children[0].exited
        assert not manager.has_session("room-a")
    finally:
        for child in children:
            child.refuse_close = False
        await manager.stop()


class ForgetfulProcess(WorkspaceProcess):
    """A native close that returns after its waits failed: the child keeps
    running and the transport forgets it, as the SDK's close can."""

    async def close(self) -> None:
        self._process = None


@asynccontextmanager
async def started(
    child_type: type[WorkspaceProcess], tmp_path: Path
) -> AsyncIterator[tuple[transport.ConfirmedCloseTransport, WorkspaceProcess]]:
    """An owned transport over a real child, reaped on exit whatever the test
    asserted."""
    native = child_type(
        prompt="",
        options=ClaudeAgentOptions(
            cli_path=sys.executable, cwd=str(tmp_path), env={"PROBE_MARKER": "close"}
        ),
    )
    owned = transport.ConfirmedCloseTransport(native)
    await owned.connect()
    child = native._child
    # The test may patch the child's own kill/wait to fail or block; reap with
    # the class's unpatched operations, and within a bound.
    kill, wait = type(child).kill, type(child).wait
    try:
        yield owned, native
    finally:
        if child.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                kill(child)
            async with asyncio.timeout(10):
                await wait(child)


async def test_a_close_that_falsely_succeeds_still_reaps_the_child(
    tmp_path: Path,
) -> None:
    async with started(ForgetfulProcess, tmp_path) as (owned, native):
        await owned.close()

        assert native.exited
        assert not owned.close_pending


async def test_an_unconfirmed_exit_keeps_the_child_for_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with started(ForgetfulProcess, tmp_path) as (owned, native):
        child = native._child
        real_kill = child.kill
        monkeypatch.setattr(
            child, "kill", lambda: (_ for _ in ()).throw(PermissionError())
        )

        with pytest.raises(transport.ClaudeProcessCloseError):
            await owned.close()
        assert owned.close_pending
        assert not native.exited

        monkeypatch.setattr(child, "kill", real_kill)
        await owned.close()

        assert native.exited
        assert not owned.close_pending


async def test_an_exit_that_never_confirms_fails_within_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transport, "CHILD_EXIT_TIMEOUT_SECONDS", 0.2)
    async with started(ForgetfulProcess, tmp_path) as (owned, native):
        child = native._child
        real_wait = child.wait

        async def never() -> int:
            await asyncio.Event().wait()
            return 0

        monkeypatch.setattr(child, "wait", never)
        async with asyncio.timeout(5):
            with pytest.raises(transport.ClaudeProcessCloseError):
                await owned.close()
        assert owned.close_pending

        monkeypatch.setattr(child, "wait", real_wait)
        await owned.close()
        assert native.exited
        assert not owned.close_pending


async def test_a_close_that_raised_keeps_the_child_for_a_retry(tmp_path: Path) -> None:
    async with started(WorkspaceProcess, tmp_path) as (owned, native):
        native.refuse_close = True

        with pytest.raises(RuntimeError, match="cleanup failed"):
            await owned.close()
        assert owned.close_pending
        assert not native.exited

        native.refuse_close = False
        await owned.close()

        assert native.exited
        assert not owned.close_pending
