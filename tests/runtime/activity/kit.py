from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx

from band.platform.event import PlatformEvent
from band.platform.link import BandLink
from band.runtime.execution import ExecutionContext
from band.runtime.runtime import AgentRuntime
from band.runtime.sessionconfig import SessionConfig
from band.runtime.tools.agent import AgentTools
from band.runtime.working_state import (
    end_tool_activity,
    report_memory_activity,
    report_tool_activity,
    set_permission_wait,
)
from tests.runtime.helpers import (
    AGENT_ID,
    ROOM_ID,
    LifecyclePlatform,
    rest_client_over,
)


class ActivityPeer(LifecyclePlatform):
    """Hold an activity response while controls or the turn's lifetime advance."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def respond(self, request: httpx.Request) -> httpx.Response:
        response = self.answer(request)
        if self.hold and request.url.path.endswith("/activity"):
            self.hold = False
            self.entered.set()
            await self.release.wait()
        return response


@asynccontextmanager
async def activity_context(
    peer: ActivityPeer,
    handler: Callable[[ExecutionContext, PlatformEvent], Awaitable[None]],
    observer: AsyncMock,
    *,
    enabled: bool = True,
    max_cycle_seconds: float | None = None,
) -> AsyncIterator[ExecutionContext]:
    async with rest_client_over(peer.respond) as rest:
        link = BandLink(agent_id=AGENT_ID, api_key="test-key")
        link.rest = rest
        runtime = AgentRuntime(
            link=link, agent_id=AGENT_ID, on_execute=handler, on_control=observer
        )
        ctx = ExecutionContext(
            ROOM_ID,
            link,
            handler,
            agent_id=AGENT_ID,
            on_platform_stop=runtime._on_platform_stop,
            config=SessionConfig(
                enable_context_hydration=False,
                enable_working_state=enabled,
                max_cycle_seconds=max_cycle_seconds,
            ),
        )
        try:
            yield ctx
        finally:
            peer.release.set()
            await ctx.stop()


class DetailPeer(ActivityPeer):
    """Capture the activity detail each report carries, alongside the base tuples."""

    def __init__(self) -> None:
        super().__init__()
        self.details: list[dict[str, object] | None] = []

    def answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/activity"):
            self.details.append(json.loads(request.content).get("activity"))
        return super().answer(request)


def reporting_handler(
    *, tool: str | None = None, permission: bool = False, memory: bool = False
):
    async def handler(ctx: ExecutionContext, event: PlatformEvent) -> None:
        tools = AgentTools.from_context(ctx)
        if permission:
            await set_permission_wait(tools, True)
            await set_permission_wait(tools, False)
        if memory:
            await report_memory_activity(tools)
        if tool is not None:
            await report_tool_activity(tools, tool)
            await end_tool_activity(tools)

    return handler
