"""ACPRuntime owns the agent process it spawned until it is confirmed closed,
and reports MCP transport support only on an explicit advertisement."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp.schema import AgentCapabilities, InitializeResponse, McpCapabilities

from band.integrations.acp.client_runtime import (
    ACPProcessCloseError,
    ACPRuntime,
    ACPRuntimeClosedError,
)


async def spawn_sleeper() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )


def connection(init: object = None) -> AsyncMock:
    conn = AsyncMock()
    conn.initialize = AsyncMock(return_value=init or MagicMock())
    return conn


class LeakySpawn:
    """A spawn context whose exit leaves the child running, like a context
    whose exit already ran once (a second exit is a no-op)."""

    def __init__(self, init: object = None) -> None:
        self.init = init
        self.processes: list[asyncio.subprocess.Process] = []

    @asynccontextmanager
    async def __call__(
        self, client: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[tuple[Any, Any]]:
        process = await spawn_sleeper()
        self.processes.append(process)
        yield connection(self.init), process


def runtime_for(spawn: Any) -> ACPRuntime:
    return ACPRuntime(
        command=["agent"], spawn_process=spawn, pass_builtin_transport_options=False
    )


async def test_stop_reaps_a_process_its_context_left_running() -> None:
    spawn = LeakySpawn()
    runtime = runtime_for(spawn)
    await runtime.start()

    await runtime.stop()

    assert spawn.processes[0].returncode is not None
    assert not runtime.close_pending


async def test_unconfirmed_close_keeps_the_process_and_blocks_respawn() -> None:
    spawn = LeakySpawn()
    runtime = runtime_for(spawn)
    await runtime.start()
    process = spawn.processes[0]
    real_wait = process.wait
    process.wait = AsyncMock(side_effect=OSError("wait failed"))  # type: ignore[method-assign]

    with pytest.raises(ACPProcessCloseError):
        await runtime.stop()

    assert runtime.close_pending
    with pytest.raises(ACPProcessCloseError):
        await runtime.ensure_connection(can_respawn=True)
    assert len(spawn.processes) == 1

    process.wait = real_wait  # type: ignore[method-assign]
    await runtime.stop()

    assert process.returncode is not None
    assert not runtime.close_pending
    await runtime.ensure_connection(can_respawn=True)
    assert len(spawn.processes) == 2
    await runtime.stop()


async def test_failed_start_reaps_the_process_it_spawned() -> None:
    spawn = LeakySpawn()
    runtime = runtime_for(spawn)

    @asynccontextmanager
    async def failing(client: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async with spawn(client) as (conn, process):
            conn.initialize = AsyncMock(side_effect=RuntimeError("init failed"))
            yield conn, process

    runtime._spawn_process = failing

    with pytest.raises(RuntimeError, match="init failed"):
        await runtime.start()

    assert spawn.processes[0].returncode is not None
    assert not runtime.close_pending


async def test_closed_runtime_never_respawns() -> None:
    spawn = LeakySpawn()
    runtime = runtime_for(spawn)
    await runtime.start()

    await runtime.close()

    with pytest.raises(ACPRuntimeClosedError):
        await runtime.ensure_connection(can_respawn=True)
    assert len(spawn.processes) == 1


def init_with(mcp: McpCapabilities | None) -> InitializeResponse:
    return InitializeResponse(
        protocol_version=1,
        agent_capabilities=AgentCapabilities(mcp_capabilities=mcp),
    )


@pytest.mark.parametrize(
    ("mcp", "http", "sse"),
    [
        (None, False, False),
        (McpCapabilities(), False, False),
        (McpCapabilities(http=None, sse=None), False, False),
        (McpCapabilities(http=False, sse=False), False, False),
        (McpCapabilities(http=True), True, False),
        (McpCapabilities(sse=True), False, True),
        (McpCapabilities(http=True, sse=True), True, True),
    ],
)
async def test_mcp_support_is_only_what_the_agent_advertises(
    mcp: McpCapabilities | None, http: bool, sse: bool
) -> None:
    runtime = runtime_for(LeakySpawn(init_with(mcp)))
    await runtime.start()

    assert (runtime.agent_mcp_http, runtime.agent_mcp_sse) == (http, sse)

    await runtime.stop()
    assert (runtime.agent_mcp_http, runtime.agent_mcp_sse) == (False, False)
