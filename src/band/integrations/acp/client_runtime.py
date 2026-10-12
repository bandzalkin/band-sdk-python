"""Connection lifecycle and session operations for outbound ACP clients."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

from acp import spawn_agent_process
from acp.schema import (
    ClientCapabilities,
)

from band.integrations.acp.collecting import ACPCollectingClient
from band.integrations.acp.permissions import (
    ALLOW_ALWAYS_KIND,
    ElicitationHandler,
    ElicitationNarrator,
    PermissionHandler,
    PermissionNarrator,
    allow_permission,
    cancel_permission,
    elicitation_requested_schema,
    elicitation_session_id,
    option_id_of_kind,
    permission_option_ids,
    select_allow_option_id,
)
from band.integrations.acp.sessions import (
    ACP_SESSION_LOAD_TIMEOUT_SECONDS,
    ACPSessionOperations,
)
from band.integrations.acp.stderr import (
    STDERR_DRAIN_TIMEOUT_S,
    STDERR_TAIL_LINES,
    ACPStderrDrain,
)
from band.integrations.acp.stream import ChunkSink
from band.integrations.acp.transport import (
    ACP_STDIO_LIMIT_BYTES,
    ACPConnectionProtocol,
    ACPSpawnContextProtocol,
    tcp_spawn_process,
)
from band.integrations.mcp.backends import BandMCPTransport

__all__ = [
    "ACP_SESSION_LOAD_TIMEOUT_SECONDS",
    "ACP_STDIO_LIMIT_BYTES",
    "ALLOW_ALWAYS_KIND",
    "STDERR_DRAIN_TIMEOUT_S",
    "STDERR_TAIL_LINES",
    "ACPCollectingClient",
    "ACPConnectionProtocol",
    "ACPProcessCloseError",
    "ACPRuntime",
    "ACPRuntimeClosedError",
    "ACPSpawnContextProtocol",
    "ChunkSink",
    "ElicitationHandler",
    "ElicitationNarrator",
    "PermissionHandler",
    "PermissionNarrator",
    "allow_permission",
    "cancel_permission",
    "elicitation_requested_schema",
    "elicitation_session_id",
    "option_id_of_kind",
    "permission_option_ids",
    "select_allow_option_id",
    "tcp_spawn_process",
]

logger = logging.getLogger(__name__)


class ACPProcessCloseError(RuntimeError):
    """The ACP agent process could not be confirmed closed; its runtime keeps it
    for a later ``stop`` to retry."""


class ACPRuntimeClosedError(RuntimeError):
    """The runtime was closed by its owner and will not spawn again."""


class ACPRuntime(ACPSessionOperations):
    """Generic ACP subprocess runtime shared by outbound ACP bridges."""

    def __init__(
        self,
        *,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        auth_method: str | None = None,
        client_factory: Callable[[], ACPCollectingClient] | None = None,
        spawn_process: Callable[..., object] | None = None,
        client_capabilities: ClientCapabilities | None = None,
        use_unstable_protocol: bool = False,
        pass_builtin_transport_options: bool = True,
    ) -> None:
        super().__init__()
        self._command = list(command)
        self._env = env
        self._cwd = cwd
        self._auth_method = auth_method
        self._client_factory = client_factory or ACPCollectingClient
        self._spawn_process = spawn_process or spawn_agent_process
        self._client_capabilities = client_capabilities
        self._use_unstable_protocol = use_unstable_protocol
        self._pass_builtin_transport_options = pass_builtin_transport_options

        self._conn: ACPConnectionProtocol | None = None
        self._ctx: (
            AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]] | None
        ) = None
        self._stop_lock = asyncio.Lock()
        # The spawned agent process, held until it is confirmed reaped.
        self._process: asyncio.subprocess.Process | None = None
        self._agent_mcp_http = False
        self._agent_mcp_sse = False
        self._stderr = ACPStderrDrain(logger)
        self._closed = False

    async def close(self) -> None:
        """Stop for good: unlike ``stop``, a closed runtime never respawns, so a
        caller still holding it cannot start a process its owner no longer
        tracks. Retry after ``ACPProcessCloseError`` to confirm the process."""
        self._closed = True
        await self.stop()

    async def start(self, *, respawn: bool = False) -> None:
        """Spawn or respawn the ACP agent subprocess."""
        async with self._stop_lock:
            await self._start(respawn=respawn)

    async def _start(self, *, respawn: bool) -> None:
        if self._closed:
            raise ACPRuntimeClosedError("the ACP runtime was closed")
        if self.close_pending:
            raise ACPProcessCloseError(
                "the previous ACP agent process is not confirmed closed"
            )
        self._connection_failed = False
        logger.info(
            "%s ACP agent subprocess",
            "Respawning" if respawn else "Spawning",
        )

        ctx = self._spawn_context()
        self._ctx = ctx
        try:
            self._conn, transport = await ctx.__aenter__()
            if isinstance(transport, asyncio.subprocess.Process):
                self._process = transport
                self._stderr.start(transport)
            await self._initialize_connection(self._conn)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self._stderr.expect_exit(connection_failed=self._connection_failed)
            await self._cleanup_failed_start(ctx, "init cancel")
            raise
        except Exception:
            await self._cleanup_failed_start(ctx, "init failure")
            raise
        # A connect-only transport (e.g. TCP) carries no command; describe it
        # rather than logging a blank suffix.
        logger.info(
            "Connected to ACP agent: %s",
            " ".join(self._command) or "<injected transport>",
        )

    def _spawn_context(
        self,
    ) -> AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]]:
        self._client = self._client_factory()  # type: ignore[abstract]  # ACP client protocol defines optional hooks as abstract
        spawn_kwargs: dict[str, Any] = {}
        if self._pass_builtin_transport_options:
            spawn_kwargs["transport_kwargs"] = {"limit": ACP_STDIO_LIMIT_BYTES}
            if self._use_unstable_protocol:
                spawn_kwargs["use_unstable_protocol"] = True
        return cast(
            AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]],
            self._spawn_process(
                self._client,
                # TCP transports carry no command; stdio carries executable and args.
                *self._command,
                env=self._env,
                cwd=self._cwd,
                **spawn_kwargs,
            ),
        )

    async def _initialize_connection(self, conn: ACPConnectionProtocol) -> None:
        init_kwargs: dict[str, Any] = {"protocol_version": 1}
        if self._client_capabilities is not None:
            init_kwargs["client_capabilities"] = self._client_capabilities
        init_response = await cast(Any, conn).initialize(**init_kwargs)
        self._agent_mcp_transport = self._select_mcp_transport(init_response)
        self._agent_mcp_http = self._advertised_mcp(init_response, "http")
        self._agent_mcp_sse = self._advertised_mcp(init_response, "sse")
        self._agent_supports_session_load = self._select_session_load(init_response)
        self._agent_supports_session_close = self._select_session_close(init_response)
        if self._auth_method:
            await conn.authenticate(method_id=self._auth_method)
            logger.info("Authenticated with method: %s", self._auth_method)

    async def ensure_connection(self, *, can_respawn: bool) -> ACPConnectionProtocol:
        async with self._stop_lock:
            if self._conn is None:
                if self.close_pending:
                    raise ACPProcessCloseError(
                        "the previous ACP agent process is not confirmed closed"
                    )
                if self._ctx is None and can_respawn:
                    await self._start(respawn=False)
                else:
                    raise RuntimeError(
                        "ACP client not initialized. Call on_started first."
                    )

            conn = self._conn

        if conn is None:
            raise RuntimeError("ACP client connection dropped before prompt")
        return conn

    async def stop(self) -> None:
        """Close the agent connection and confirm its process exited.

        Raises ``ACPProcessCloseError`` when the process cannot be confirmed
        closed; the runtime then keeps the process handle (``close_pending``) and
        a later ``stop`` retries the confirmation.
        """
        ctx: AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]] | None
        async with self._stop_lock:
            self._stderr.expect_exit(connection_failed=self._connection_failed)
            ctx = self._ctx
            self._ctx = None
            self._conn = None
            self._client = None
            self._agent_supports_session_load = False
            self._agent_supports_session_close = False
            self._agent_mcp_http = False
            self._agent_mcp_sse = False
            try:
                if ctx is not None:
                    await ctx.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error during ACP runtime shutdown")
            finally:
                # Confirm the child before draining its stderr: a child the
                # context left running keeps the pipe open.
                try:
                    await self._confirm_process_exit()
                finally:
                    await self._stderr.finish()

    async def _cleanup_failed_start(
        self,
        ctx: AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]],
        reason: str,
    ) -> None:
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.exception("Error cleaning up ACP subprocess after %s", reason)
        finally:
            try:
                await self._confirm_process_exit()
            except ACPProcessCloseError:
                # The start's own error is what the caller reports; the
                # unconfirmed process stays owned (``close_pending``) for a
                # later stop.
                logger.exception("ACP subprocess not confirmed closed after %s", reason)
            finally:
                await self._stderr.finish()
        self._ctx = None
        self._conn = None
        self._agent_supports_session_load = False
        self._agent_supports_session_close = False
        self._agent_mcp_http = False
        self._agent_mcp_sse = False

    async def _confirm_process_exit(self) -> None:
        """Reap the agent process this runtime spawned, or raise and keep it.

        The spawn context is an async generator: once its exit has run (even if it
        raised), exiting it again is a no-op that proves nothing about the child.
        Ownership is therefore the process handle itself, released only once the
        child is reaped, so a failed close can be retried and the process is never
        silently abandoned. Connect-only transports carry no process to confirm.
        """
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            try:
                await process.wait()
            except Exception as error:
                raise ACPProcessCloseError(
                    "the ACP agent process could not be confirmed closed"
                ) from error
        self._process = None

    @property
    def close_pending(self) -> bool:
        """Whether an earlier stop could not confirm the agent process closed;
        the runtime refuses to start again until a later ``stop`` succeeds."""
        return self._process is not None and self._ctx is None

    @property
    def closed(self) -> bool:
        """Whether ``close`` was called: the runtime never spawns again."""
        return self._closed

    @property
    def agent_mcp_http(self) -> bool:
        """The connected agent explicitly advertised HTTP MCP servers."""
        return self._agent_mcp_http

    @property
    def agent_mcp_sse(self) -> bool:
        """The connected agent explicitly advertised SSE MCP servers."""
        return self._agent_mcp_sse

    @staticmethod
    def _select_mcp_transport(init_response: object) -> BandMCPTransport:
        capabilities = getattr(init_response, "agent_capabilities", None)
        mcp_capabilities = getattr(capabilities, "mcp_capabilities", None)

        if getattr(mcp_capabilities, "http", False):
            return BandMCPTransport.HTTP
        if getattr(mcp_capabilities, "sse", False):
            return BandMCPTransport.SSE

        return BandMCPTransport.HTTP

    @staticmethod
    def _advertised_mcp(init_response: object, transport: str) -> bool:
        """Only an explicit ``true`` is support: missing, null or false is not, and
        the preferred Band transport is not evidence either way."""
        capabilities = getattr(init_response, "agent_capabilities", None)
        mcp_capabilities = getattr(capabilities, "mcp_capabilities", None)
        return getattr(mcp_capabilities, transport, None) is True

    @staticmethod
    def _select_session_load(init_response: object) -> bool:
        capabilities = getattr(init_response, "agent_capabilities", None)
        return getattr(capabilities, "load_session", False) is True

    @staticmethod
    def _select_session_close(init_response: object) -> bool:
        capabilities = getattr(init_response, "agent_capabilities", None)
        session_capabilities = getattr(capabilities, "session_capabilities", None)
        return getattr(session_capabilities, "close", None) is not None
