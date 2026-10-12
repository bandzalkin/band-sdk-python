"""
Session manager for Claude Agent SDK clients.

Maintains one ClaudeSDKClient instance per Band chat room to ensure
conversation continuity within each room.

Uses a dedicated background task for all session operations to ensure
connect() and disconnect() are always called from the same task context.

A new connection is owned from the moment the queue reserves it: one
``RoomStartup`` per room holds its waiters, the worker that resolves its MCP
servers outside the queue, and whatever it has launched. The queue only
reserves, connects and disconnects; it never awaits a worker. Cancellation,
room cleanup and ``stop()`` retire the reservation, and its launched child,
file and workspace claim stay owned until that retirement settles.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from band.integrations.claude_sdk.mcp_startup import (
    ClaudeMcpStartupError,
    McpConfigFile,
    McpServerResolver,
    await_mcp_ready,
    freeze_external_servers,
    startup_error,
)
from band.workspaces import RoomWorkspaces

try:
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        ClaudeAgentOptions,
        ClaudeSDKClient,
    )
    from claude_agent_sdk._internal.transport import Transport
    from claude_agent_sdk.types import (  # type: ignore[import-not-found]
        CanUseTool,
        McpServerConfig,
    )

    from band.integrations.claude_sdk.transport import create_transport

    _CLAUDE_SDK_AVAILABLE = True
except ImportError:
    _CLAUDE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClaudeSessionManagerStoppedError(RuntimeError):
    """A session was requested from a manager that ``stop()`` shut down."""


class _Retry:
    """The room was retiring when the queue saw the request; ask again."""


_RETRY = _Retry()


@dataclass
class SessionCommand:
    """Command to be processed by the session manager task."""

    action: str  # "create", "cleanup", "cleanup_all", "invalidate", "call", "stop"
    room_id: str | None = None
    resume_session_id: str | None = None
    result_future: asyncio.Future[Any] | None = None
    release_workspace: bool = True
    call: Callable[[], Awaitable[Any]] | None = None


@dataclass(eq=False)
class RoomStartup:
    """One room's not-yet-published connection and everything it owns."""

    room_id: str
    resume_session_id: str | None
    protected: Any
    result: asyncio.Future[ClaudeSDKClient]
    waiters: int = 0
    # Retired: it never launches, publishes or serves a waiter afterwards.
    closing: bool = False
    # Its worker's cancellation is under way (requested here or by the worker
    # itself), so no owner injects another one into its settlement.
    cancel_requested: bool = False
    worker: asyncio.Task[None] | None = None
    client: ClaudeSDKClient | None = None
    transport: Transport | None = None
    config_file: McpConfigFile | None = None

    def retire(self) -> None:
        """Close admission and see the worker cancelled at most once; a worker
        retiring itself, or one already being cancelled, is never cancelled."""
        self.closing = True
        worker = self.worker
        if worker is None or worker.done() or self.cancel_requested:
            return
        self.cancel_requested = True
        if worker is asyncio.current_task() or worker.cancelling():
            return
        worker.cancel()


@dataclass(eq=False)
class _Retirement:
    task: asyncio.Task[None]
    startup: RoomStartup | None = field(default=None)
    release_workspace: bool = True


class _StartupRetired(Exception):
    """The startup was retired while its worker ran."""


class _StartupReplaced(Exception):
    """The protected servers changed while the startup ran; retry on the new ones."""


class ClaudeSessionManager:
    """
    Manages ClaudeSDKClient instances per chat room.

    Each room gets its own ClaudeSDKClient instance to maintain separate
    conversation histories and sessions.

    Key features:
    - Lazy initialization (clients created on first message)
    - Session reuse (same client for all messages in a room)
    - Graceful cleanup on room leave/disconnect
    - All session operations run in a single background task to avoid
      cross-task asyncio issues with connect/disconnect

    Example:
        import logging
        logger = logging.getLogger(__name__)

        manager = ClaudeSessionManager(base_options)
        await manager.start()  # Start background task

        # Get client for room (creates if doesn't exist)
        client = await manager.get_or_create_session("room-123")

        # Use client
        await client.query("Hello")
        async for msg in client.receive_response():
            logger.info("%s", msg)

        # Cleanup when done
        await manager.cleanup_session("room-123")
        await manager.stop()  # Stop background task
    """

    def __init__(
        self,
        base_options: ClaudeAgentOptions,
        can_use_tool_factory: Callable[[str], CanUseTool] | None = None,
        mcp_servers_factory: Callable[[str], dict[str, McpServerConfig]] | None = None,
        workspaces: RoomWorkspaces | None = None,
        resolve_mcp_servers: McpServerResolver | None = None,
    ) -> None:
        """
        Initialize session manager.

        Args:
            base_options: Base ClaudeAgentOptions to use for all clients.
                          These options are shared across all room sessions.
            can_use_tool_factory: Optional factory that creates a room-specific
                ``can_use_tool`` callback.  When set, each new session receives
                its own callback bound to the room_id.
            mcp_servers_factory: Optional factory that returns a room's
                ``mcp_servers``.  When set, it replaces
                ``base_options.mcp_servers`` for each new session. With
                ``resolve_mcp_servers`` it supplies only the protected (Band)
                servers, merged after the resolved ones.
            workspaces: Adapter-local claims; omit to preserve base_options.cwd.
            resolve_mcp_servers: Returns the complete external MCP servers for a
                new connection's workspace. Called outside the session queue,
                once per new connection; reuse never calls it. Its servers and
                the protected ones are launched from a private file, and the
                session is published only once every one reports connected.
        """
        self.base_options = base_options
        self._can_use_tool_factory = can_use_tool_factory
        self._mcp_servers_factory = mcp_servers_factory
        self._resolve_mcp_servers = resolve_mcp_servers
        self._sessions: dict[str, ClaudeSDKClient] = {}
        self._transports: dict[str, Transport] = {}
        self._config_files: dict[str, McpConfigFile] = {}
        # The protected servers each published session dials (callback mode).
        self._identities: dict[str, Any] = {}
        self._unusable: set[str] = set()
        self._workspaces = workspaces
        self._startups: dict[str, RoomStartup] = {}
        self._retirements: dict[str, _Retirement] = {}
        self._command_queue: asyncio.Queue[SessionCommand] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None
        self._closing = False
        logger.info("ClaudeSessionManager initialized")

    @property
    def _loop_running(self) -> bool:
        return self._task is not None

    def is_startup_task(self, task: asyncio.Task[Any] | None) -> bool:
        """Whether ``task`` is a session startup this manager's teardown waits
        for; such a task must request teardown without waiting on it."""
        return task is not None and any(
            startup.worker is task for startup in self._startups.values()
        )

    async def start(self) -> None:
        """Start the background task that manages all sessions."""
        if self._shutdown is not None or self._closing:
            raise ClaudeSessionManagerStoppedError()
        if self._loop_running:
            return

        self._task = asyncio.create_task(self._run_session_loop())
        logger.info("ClaudeSessionManager background task started")

    async def stop(self) -> None:
        """Stop the background task and cleanup all sessions, for good: the
        adapter builds a new manager when it starts again. Every caller
        awaits the same shutdown, which a cancelled caller doesn't abandon.

        Admission closes at once. A session startup that asks for the stop
        from inside itself returns without waiting for the shutdown, which
        waits for that startup to settle."""
        self._closing = True
        for startup in list(self._startups.values()):
            startup.retire()
        if self._shutdown is None or (
            self._shutdown.done() and self._shutdown.exception() is not None
        ):
            self._shutdown = asyncio.create_task(self._shut_down())
        if self.is_startup_task(asyncio.current_task()):
            return
        await asyncio.shield(self._shutdown)

    async def _shut_down(self) -> None:
        await self._settle_startups(list(self._startups.values()))
        for retirement in list(self._retirements.values()):
            await asyncio.wait((retirement.task,))
        if self._task is None:
            return
        if not self._task.done():
            result = asyncio.get_running_loop().create_future()
            self._command_queue.put_nowait(
                SessionCommand(action="stop", result_future=result)
            )
            await result
            await self._task
        self._task = None
        self._fail_pending_commands()
        logger.info("ClaudeSessionManager background task stopped")

    def _fail_pending_commands(self) -> None:
        """Settle every command still queued once the loop has exited, so no
        caller waits forever on a future nothing will resolve. Teardown
        commands succeed (stop already tore every session down); only a
        ``create`` or a ``call`` can't be honoured."""
        while not self._command_queue.empty():
            cmd = self._command_queue.get_nowait()
            if cmd.result_future is None or cmd.result_future.done():
                continue
            if cmd.action in ("create", "call"):
                cmd.result_future.set_exception(ClaudeSessionManagerStoppedError())
            else:
                cmd.result_future.set_result(None)

    async def _run_session_loop(self) -> None:
        """Background task that processes all session commands."""
        logger.debug("Session loop started")

        while True:
            cmd: SessionCommand | None = None
            try:
                cmd = await self._command_queue.get()

                if cmd.action == "create":
                    if cmd.result_future is not None and cmd.result_future.done():
                        # Its caller left before the queue reached it.
                        self._command_queue.task_done()
                        continue
                    outcome = await self._do_reserve(cmd.room_id, cmd.resume_session_id)
                    self._settle(cmd, outcome)

                elif cmd.action == "call":
                    assert cmd.call is not None
                    self._settle(cmd, await cmd.call())

                elif cmd.action == "cleanup":
                    await self._do_cleanup_session(
                        cmd.room_id, release_workspace=cmd.release_workspace
                    )
                    self._settle(cmd, None)

                elif cmd.action == "invalidate":
                    await self._do_invalidate_session(cmd.room_id)
                    self._settle(cmd, None)

                elif cmd.action == "cleanup_all":
                    await self._do_cleanup_all()
                    self._settle(cmd, None)

                elif cmd.action == "stop":
                    await self._do_cleanup_all()
                    self._settle(cmd, None)
                    break

                self._command_queue.task_done()

            except asyncio.CancelledError:
                logger.debug("Session loop cancelled")
                break
            except Exception as e:
                if isinstance(e, _StartupRetired | _StartupReplaced):
                    pass
                elif isinstance(e, ClaudeMcpStartupError):
                    logger.error("Session command failed: %s", e)
                else:
                    logger.exception("Error in session loop")
                if cmd and cmd.result_future and not cmd.result_future.done():
                    cmd.result_future.set_exception(e)

        logger.debug("Session loop exited")

    @staticmethod
    def _settle(cmd: SessionCommand, outcome: Any) -> None:
        if cmd.result_future is not None and not cmd.result_future.done():
            cmd.result_future.set_result(outcome)

    def _build_options(
        self,
        room_id: str,
        resume_session_id: str | None = None,
        *,
        mcp_servers: Any = None,
    ) -> ClaudeAgentOptions:
        """Build ``ClaudeAgentOptions`` for a specific room session.

        Uses ``dataclasses.replace()`` to shallow-copy all fields from
        ``base_options``, then applies room-specific overrides.
        """
        overrides: dict[str, Any] = {}
        if self._workspaces is not None:
            overrides["cwd"] = self._workspaces.claim(room_id)

        if resume_session_id:
            overrides["resume"] = resume_session_id

        if self._can_use_tool_factory:
            overrides["can_use_tool"] = self._can_use_tool_factory(room_id)

        if mcp_servers is not None:
            overrides["mcp_servers"] = mcp_servers
        elif self._mcp_servers_factory:
            overrides["mcp_servers"] = self._mcp_servers_factory(room_id)

        return dataclasses.replace(self.base_options, **overrides)

    # -- reservation (queue) -------------------------------------------------

    async def _do_reserve(
        self, room_id: str | None, resume_session_id: str | None
    ) -> ClaudeSDKClient | RoomStartup | _Retry:
        """Reuse the room's client, join its startup, or reserve a new one.
        Runs in the queue and never waits for a startup's worker."""
        if not room_id:
            raise ValueError("room_id is required")
        if self._closing:
            raise ClaudeSessionManagerStoppedError()
        if room_id in self._retirements:
            return _RETRY

        if (client := self._sessions.get(room_id)) is not None:
            if room_id not in self._unusable and self._is_current(room_id, client):
                logger.debug("Reusing existing session for room: %s", room_id)
                return client
            logger.info("Replacing unusable or outdated session for room %s", room_id)
            await self._do_cleanup_session(room_id, release_workspace=False)

        if (startup := self._startups.get(room_id)) is not None:
            if startup.closing:
                return _RETRY
            startup.waiters += 1
            return startup

        if resume_session_id:
            logger.info("Resuming session %s for room: %s", resume_session_id, room_id)
        else:
            logger.info("Creating new ClaudeSDKClient session for room: %s", room_id)
        if self._workspaces is not None:
            self._workspaces.claim(room_id)
        startup = RoomStartup(
            room_id=room_id,
            resume_session_id=resume_session_id,
            protected=self._protected_identity(room_id),
            result=asyncio.get_running_loop().create_future(),
            waiters=1,
        )
        self._startups[room_id] = startup
        startup.worker = asyncio.create_task(
            self._run_startup(startup), name=f"claude-startup:{room_id}"
        )
        return startup

    def _protected_identity(self, room_id: str) -> Any:
        if self._resolve_mcp_servers is None or self._mcp_servers_factory is None:
            return None
        return self._mcp_servers_factory(room_id)

    def _is_current(self, room_id: str, client: ClaudeSDKClient) -> bool:
        """Whether ``client`` still dials the MCP servers the room has now.

        With a resolver, only the retained protected servers are compared:
        reuse never re-resolves the external ones."""
        if self._mcp_servers_factory is None:
            return True
        if self._resolve_mcp_servers is not None:
            return self._identities.get(room_id) == self._mcp_servers_factory(room_id)
        return client.options.mcp_servers == self._mcp_servers_factory(room_id)

    # -- startup worker (outside the queue) ----------------------------------

    def _require_live(self, startup: RoomStartup) -> None:
        if (
            startup.closing
            or self._closing
            or self._startups.get(startup.room_id) is not startup
            or startup.waiters <= 0
        ):
            raise _StartupRetired
        if startup.protected != self._protected_identity(startup.room_id):
            # The Band server it would dial was replaced: never publish against
            # the old endpoint; its waiters start again on the current one.
            raise _StartupReplaced

    async def _in_queue(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """Run ``call`` on the manager's own task, where connect and disconnect
        happen. Shielded: the queue finishes what it starts."""
        result: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        # A caller cancelled meanwhile no longer reads the outcome; its owner
        # (the startup's discard path) handles it, so never warn about it.
        result.add_done_callback(lambda done: done.cancelled() or done.exception())
        await self._command_queue.put(
            SessionCommand(action="call", result_future=result, call=call)
        )
        return await asyncio.shield(result)

    async def _run_startup(self, startup: RoomStartup) -> None:
        room_id = startup.room_id
        try:
            external: dict[str, Any] | None = None
            if self._resolve_mcp_servers is not None:
                workspace = self._startup_workspace(room_id)
                try:
                    resolved = await self._resolve_mcp_servers(workspace)
                except Exception as error:  # noqa: BLE001 -- any callback failure is a typed startup failure
                    raise startup_error(
                        "resolution", "resolve_mcp_servers failed", error
                    ) from None
                external = freeze_external_servers(resolved, startup.protected or {})
            self._require_live(startup)
            await self._in_queue(lambda: self._do_launch(startup, external))
            if external is not None:
                client = startup.client
                assert client is not None
                expected = frozenset({*external, *(startup.protected or {})})
                await await_mcp_ready(client.get_mcp_status, expected)
            self._require_live(startup)
            self._publish(startup)
        except BaseException as error:
            await self._discard_startup(startup, error)
            if isinstance(error, asyncio.CancelledError) and not startup.closing:
                raise

    def _startup_workspace(self, room_id: str) -> str:
        if self._workspaces is not None:
            return self._workspaces.claim(room_id)
        return str(self.base_options.cwd or Path.cwd())

    async def _do_launch(
        self, startup: RoomStartup, external: dict[str, Any] | None
    ) -> None:
        """Create and connect the startup's client (in the queue)."""
        self._require_live(startup)
        room_id = startup.room_id
        mcp_servers: Any = None
        if external is not None:
            startup.config_file = McpConfigFile.create(
                {**external, **(startup.protected or {})}
            )
            mcp_servers = str(startup.config_file.path)
        options = self._build_options(
            room_id, startup.resume_session_id, mcp_servers=mcp_servers
        )
        transport = create_transport(options)
        startup.transport = transport
        startup.client = ClaudeSDKClient(options=options, transport=transport)
        try:
            await startup.client.connect()
        except Exception as error:
            if external is not None:
                raise startup_error(
                    "launch", "the Claude CLI did not start", error
                ) from None
            raise
        self._require_live(startup)

    def _publish(self, startup: RoomStartup) -> None:
        room_id = startup.room_id
        assert startup.client is not None and startup.transport is not None
        del self._startups[room_id]
        self._sessions[room_id] = startup.client
        self._transports[room_id] = startup.transport
        if startup.config_file is not None:
            self._config_files[room_id] = startup.config_file
        if startup.protected is not None:
            self._identities[room_id] = startup.protected
        startup.client = None
        startup.transport = None
        startup.config_file = None
        logger.info(
            "Session created for room %s (total sessions: %s)",
            room_id,
            len(self._sessions),
        )
        if not startup.result.done():
            startup.result.set_result(self._sessions[room_id])

    async def _discard_startup(
        self, startup: RoomStartup, error: BaseException
    ) -> None:
        """Close what the startup launched and hand its waiters the outcome.

        A close that fails keeps the client, child and file as the room's
        unusable session, so a later cleanup retries them."""
        if startup.client is not None or startup.config_file is not None:
            try:
                await self._in_queue(lambda: self._do_close_startup(startup))
            except BaseException:  # noqa: BLE001 -- custody is kept; the waiters still get the outcome
                logger.error(
                    "Room %s: could not close a failed Claude startup", startup.room_id
                )
        if self._startups.get(startup.room_id) is startup:
            del self._startups[startup.room_id]
        if not startup.result.done():
            if isinstance(error, _StartupReplaced):
                startup.result.set_exception(_StartupReplaced())
            elif isinstance(error, _StartupRetired | asyncio.CancelledError):
                startup.result.set_exception(
                    ClaudeSessionManagerStoppedError()
                    if self._closing
                    else ClaudeMcpStartupError(
                        "retirement", "the room's startup was retired"
                    )
                )
            else:
                startup.result.set_exception(error)
            # No waiter may be left to read it; mark the outcome retrieved.
            startup.result.exception()

    async def _do_close_startup(self, startup: RoomStartup) -> None:
        """Close an unpublished client (in the queue), keeping custody on failure."""
        room_id = startup.room_id
        client, transport = startup.client, startup.transport
        try:
            if client is not None:
                try:
                    await client.disconnect()
                finally:
                    if transport is not None:
                        await transport.close()
        except BaseException:
            if client is not None and room_id not in self._sessions:
                self._sessions[room_id] = client
                if transport is not None:
                    self._transports[room_id] = transport
                if startup.config_file is not None:
                    self._config_files[room_id] = startup.config_file
                self._unusable.add(room_id)
                startup.client = startup.transport = startup.config_file = None
            raise
        startup.client = startup.transport = None
        if startup.config_file is not None:
            startup.config_file.release()
            startup.config_file = None

    # -- retirement -----------------------------------------------------------

    async def _settle_startups(self, startups: list[RoomStartup]) -> None:
        for startup in startups:
            startup.retire()
        workers = [
            s.worker
            for s in startups
            if s.worker is not None and s.worker is not asyncio.current_task()
        ]
        if workers:
            await asyncio.wait(workers)

    def _leave(self, startup: RoomStartup) -> None:
        """One waiter stopped waiting; the last one retires an unsettled startup
        (the room keeps it, and its claim, until it settles)."""
        startup.waiters -= 1
        if startup.waiters <= 0 and not startup.result.done():
            self._begin_room_retirement(startup.room_id, release_workspace=False)

    def _begin_room_retirement(
        self, room_id: str, *, release_workspace: bool
    ) -> _Retirement:
        """Close the room's admission now; settle its startup and clean it up
        in one durable task that every caller joins."""
        if (retirement := self._retirements.get(room_id)) is not None:
            return retirement
        startup = self._startups.get(room_id)
        if startup is not None:
            startup.retire()
        task = asyncio.create_task(
            self._retire_room(room_id, startup, release_workspace),
            name=f"claude-retire:{room_id}",
        )
        retirement = _Retirement(
            task=task, startup=startup, release_workspace=release_workspace
        )
        self._retirements[room_id] = retirement

        def finished(done: asyncio.Task[None]) -> None:
            if self._retirements.get(room_id) is retirement:
                del self._retirements[room_id]
            if not done.cancelled() and done.exception() is not None:
                logger.warning("Room %s: Claude session cleanup failed", room_id)

        task.add_done_callback(finished)
        return retirement

    async def _retire_room(
        self, room_id: str, startup: RoomStartup | None, release_workspace: bool
    ) -> None:
        if startup is not None:
            await self._settle_startups([startup])
        if self._task is None or self._task.done():
            return
        result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._command_queue.put(
            SessionCommand(
                action="cleanup",
                room_id=room_id,
                release_workspace=release_workspace,
                result_future=result,
            )
        )
        await result

    # -- native custody (queue) ----------------------------------------------

    async def _do_invalidate_session(self, room_id: str | None) -> None:
        """Evict an exited CLI while retaining the room's workspace."""
        if not room_id:
            return
        if (transport := self._transports.get(room_id)) is not None:
            self._unusable.add(room_id)
            await transport.close()
            del self._transports[room_id]
        self._sessions.pop(room_id, None)
        self._identities.pop(room_id, None)
        self._release_config_file(room_id)
        self._unusable.discard(room_id)

    def _release_config_file(self, room_id: str) -> None:
        if (config_file := self._config_files.pop(room_id, None)) is not None:
            config_file.release()

    async def _do_cleanup_session(
        self, room_id: str | None, *, release_workspace: bool = True
    ) -> None:
        """Retain failed cleanup handles, files and claims so teardown can be
        retried."""
        if not room_id:
            return
        if (client := self._sessions.get(room_id)) is not None:
            self._unusable.add(room_id)
            try:
                await client.disconnect()
            finally:
                if (transport := self._transports.get(room_id)) is not None:
                    await transport.close()
            del self._sessions[room_id]
            self._transports.pop(room_id, None)
            self._identities.pop(room_id, None)
            self._release_config_file(room_id)
            self._unusable.discard(room_id)
        if (
            release_workspace
            and self._workspaces is not None
            and room_id not in self._startups
        ):
            self._workspaces.release(room_id)

    async def _do_cleanup_all(self) -> None:
        """Attempt every room, reporting failures without discarding ownership."""
        rooms = dict.fromkeys(self._sessions)
        if self._workspaces is not None:
            rooms.update(dict.fromkeys(self._workspaces.rooms))
        errors: list[Exception] = []
        for room_id in rooms:
            try:
                await self._do_cleanup_session(room_id)
            except Exception as error:  # noqa: BLE001 -- every room needs a cleanup attempt
                errors.append(error)
        if errors:
            raise ExceptionGroup("Claude session cleanup failed", errors)

    # -- public API ----------------------------------------------------------

    async def get_or_create_session(
        self, room_id: str, resume_session_id: str | None = None
    ) -> ClaudeSDKClient:
        """
        Get existing ClaudeSDKClient for room or create new one.

        Calls for the same room_id return the same client instance while its
        MCP servers are current; once they change (a replaced Band MCP
        server), the client is replaced, resuming ``resume_session_id``.
        Concurrent callers for a room share one startup; it is retired when
        its last caller stops waiting, and a room being cleaned up admits no
        new startup until that cleanup settles.

        Args:
            room_id: Band chat room ID (UUID)
            resume_session_id: Optional session ID to resume from a previous
                              session. Only used when creating a new session.

        Returns:
            ClaudeSDKClient instance for this room
        """
        while True:
            if self._closing:
                raise ClaudeSessionManagerStoppedError()
            if (retirement := self._retirements.get(room_id)) is not None:
                await asyncio.wait((retirement.task,))
                continue
            await self.start()
            reserved: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            await self._command_queue.put(
                SessionCommand(
                    action="create",
                    room_id=room_id,
                    resume_session_id=resume_session_id,
                    result_future=reserved,
                )
            )
            try:
                outcome = await reserved
            except asyncio.CancelledError:
                if reserved.done() and not reserved.cancelled():
                    if isinstance(outcome := reserved.result(), RoomStartup):
                        self._leave(outcome)
                else:
                    # The queue skips a create whose caller already left.
                    reserved.cancel()
                raise
            if isinstance(outcome, _Retry):
                continue
            if not isinstance(outcome, RoomStartup):
                return outcome
            try:
                return await asyncio.shield(outcome.result)
            except _StartupReplaced:
                continue
            finally:
                self._leave(outcome)

    async def cleanup_session(
        self, room_id: str, *, release_workspace: bool = True
    ) -> None:
        """
        Disconnect and remove session for a room.

        This should be called when:
        - Agent is removed from the room
        - Room is deleted
        - Adapter is shutting down

        An unsettled startup for the room is retired first; its child, file
        and claim are released only once it settles. A startup that asks for
        its own room's cleanup returns without waiting on itself.

        Args:
            room_id: Band chat room ID
            release_workspace: False when replacing a client within the same membership.
        """
        if not self._loop_running:
            return
        startup = self._startups.get(room_id)
        self_requested = (
            startup is not None and startup.worker is asyncio.current_task()
        )
        while True:
            retirement = self._begin_room_retirement(
                room_id, release_workspace=release_workspace
            )
            if self_requested:
                return
            await asyncio.shield(retirement.task)
            if retirement.release_workspace or not release_workspace:
                return
            # It joined a retirement that kept the room's claim (an abandoned
            # startup); this cleanup also releases it.

    async def invalidate_session(self, room_id: str) -> None:
        """
        Evict a dead session without calling disconnect().

        Use this when the CLI subprocess has already exited — calling
        disconnect() on a dead process would raise or hang.  After
        invalidation, ``get_or_create_session`` will create a fresh client.

        Args:
            room_id: Band chat room ID
        """
        if not self._loop_running:
            return

        result_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._command_queue.put(
            SessionCommand(
                action="invalidate",
                room_id=room_id,
                result_future=result_future,
            )
        )
        await result_future

    async def cleanup_all(self) -> None:
        """
        Disconnect all sessions.

        This should be called when the adapter is shutting down to ensure
        all Claude SDK clients are properly disconnected.
        """
        if not self._loop_running:
            return
        await self._settle_startups(list(self._startups.values()))

        result_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._command_queue.put(
            SessionCommand(
                action="cleanup_all",
                result_future=result_future,
            )
        )
        await result_future

    def has_session(self, room_id: str) -> bool:
        """Check if session exists for room."""
        return room_id in self._sessions

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_active_rooms(self) -> list[str]:
        """Get list of room IDs with active sessions."""
        return list(self._sessions.keys())

    def owns_resources(self) -> bool:
        """Whether any session, startup, file or workspace claim is still held."""
        return bool(
            self._sessions
            or self._startups
            or self._config_files
            or self._retirements
            or (self._workspaces is not None and self._workspaces.rooms)
        )
