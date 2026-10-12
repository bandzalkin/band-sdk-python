"""Connection-time MCP servers for Claude sessions: one frozen snapshot per new
connection, written to a private file, and positive per-server readiness.

Nothing here renders callback values, native status payloads or native error
text: they can carry credentials. Errors name the stage, a known server name,
a fixed reason and the exception type only.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

McpServerResolver = Callable[[str], Awaitable[Mapping[str, Any]]]
"""``resolve_mcp_servers(workspace)``: the complete external MCP servers for a
new Claude connection in ``workspace``, as native ``McpServerConfig`` values."""

MCP_READINESS_TIMEOUT_SECONDS = 30.0
_STATUS_POLL_SECONDS = 0.25
_REMOTE_TYPES = frozenset({"http", "sse"})
_FAILED_STATUSES = frozenset({"failed", "needs-auth", "disabled"})
_KNOWN_STATUSES = _FAILED_STATUSES | {"connected", "pending"}


class ClaudeMcpStartupError(RuntimeError):
    """A Claude session's MCP servers could not be resolved, launched or
    confirmed connected. Carries no callback, native status or error text."""

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        server: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.server = server
        self.error_type = error_type
        parts = [f"Claude MCP startup failed at {stage}: {reason}"]
        if server is not None:
            parts.append(f"server {server!r}")
        if error_type is not None:
            parts.append(f"({error_type})")
        super().__init__(" ".join(parts))


def startup_error(
    stage: str,
    reason: str,
    error: BaseException | None = None,
    *,
    server: str | None = None,
) -> ClaudeMcpStartupError:
    """A safe error for ``error``; raise it ``from None`` so its text, which may
    hold credentials, never joins the chain."""
    if isinstance(error, ClaudeMcpStartupError):
        return error
    return ClaudeMcpStartupError(
        stage,
        reason,
        server=server,
        error_type=None if error is None else type(error).__name__,
    )


def freeze_external_servers(
    resolved: object, protected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """An owned deep copy of the callback's servers, validated for the native CLI.

    Later changes to the mappings, lists or headers the callback returned can't
    reach the launch. In-process ``sdk`` servers and names Band reserves are
    refused by name.
    """
    if not isinstance(resolved, Mapping):
        raise ClaudeMcpStartupError(
            "resolution", "resolve_mcp_servers must return a mapping of servers"
        )
    frozen: dict[str, dict[str, Any]] = {}
    for name, config in resolved.items():
        if not isinstance(name, str) or not name:
            raise ClaudeMcpStartupError("resolution", "server names must be strings")
        if name in protected:
            raise ClaudeMcpStartupError(
                "resolution", "the name is reserved for Band", server=name
            )
        if not isinstance(config, Mapping):
            raise ClaudeMcpStartupError(
                "resolution", "server config must be a mapping", server=name
            )
        kind = config.get("type", "stdio")
        if kind == "sdk":
            raise ClaudeMcpStartupError(
                "resolution", "in-process sdk servers are not supported", server=name
            )
        if kind not in _REMOTE_TYPES and kind != "stdio":
            raise ClaudeMcpStartupError(
                "resolution", "unsupported server type", server=name
            )
        required = "command" if kind == "stdio" else "url"
        if not isinstance(config.get(required), str):
            raise ClaudeMcpStartupError(
                "resolution", f"server config needs a string {required}", server=name
            )
        try:
            frozen[name] = json.loads(json.dumps(dict(config)))
        except (TypeError, ValueError) as error:
            raise startup_error(
                "resolution", "server config is not serializable", error, server=name
            ) from None
    return frozen


@dataclass
class McpConfigFile:
    """The private native ``--mcp-config`` file one connection launches with.

    Owned until its child's exit is confirmed; then ``release`` removes it.
    """

    directory: Path
    path: Path

    @classmethod
    def create(cls, servers: Mapping[str, Any]) -> McpConfigFile:
        directory = Path(tempfile.mkdtemp(prefix="band-claude-mcp-"))
        os.chmod(directory, 0o700)
        path = directory / "mcp.json"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"mcpServers": dict(servers)}, handle)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return cls(directory=directory, path=path)

    def release(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


def _check_status(
    response: object, expected: frozenset[str]
) -> tuple[bool, str | None]:
    """``(ready, waiting_on)``; raises when the evidence is unusable or a
    server failed. Only a unique ``connected`` entry per expected name counts."""
    servers = response.get("mcpServers") if isinstance(response, Mapping) else None
    if not isinstance(servers, list):
        raise ClaudeMcpStartupError("status", "malformed status response")
    seen: dict[str, str] = {}
    for entry in servers:
        if not isinstance(entry, Mapping):
            raise ClaudeMcpStartupError("status", "malformed status entry")
        name, status = entry.get("name"), entry.get("status")
        if not isinstance(name, str) or not isinstance(status, str):
            raise ClaudeMcpStartupError("status", "malformed status entry")
        if name not in expected:
            continue
        if name in seen:
            raise ClaudeMcpStartupError("status", "duplicate status", server=name)
        if status not in _KNOWN_STATUSES:
            raise ClaudeMcpStartupError("status", "unknown status", server=name)
        if status in _FAILED_STATUSES:
            raise ClaudeMcpStartupError("status", f"server is {status}", server=name)
        seen[name] = status
    for name in sorted(expected):
        if seen.get(name) != "connected":
            return False, name
    return True, None


async def await_mcp_ready(
    get_status: Callable[[], Awaitable[object]],
    expected: frozenset[str],
    *,
    timeout: float = MCP_READINESS_TIMEOUT_SECONDS,
) -> None:
    """Wait until every expected server reports ``connected``, within one total
    deadline that also bounds each status request."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    waiting_on: str | None = None
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ClaudeMcpStartupError(
                "status", "not connected before the deadline", server=waiting_on
            )
        try:
            async with asyncio.timeout(remaining):
                response = await get_status()
        except TimeoutError:
            raise ClaudeMcpStartupError(
                "status", "not connected before the deadline", server=waiting_on
            ) from None
        except Exception as error:  # noqa: BLE001 -- any native status failure is a typed startup failure
            raise startup_error("status", "status request failed", error) from None
        ready, waiting_on = _check_status(response, expected)
        if ready:
            return
        await asyncio.sleep(min(_STATUS_POLL_SECONDS, max(0.0, deadline - loop.time())))
