from __future__ import annotations

import asyncio
import json
from itertools import groupby
from typing import Any, Literal

import httpx

from band.core.turn_lifecycle import TurnTask, bind_turn_task
from band.platform.event import PlatformEvent
from band.runtime.execution import ExecutionContext
from band.runtime.tools.agent import AgentTools
from tests.runtime.helpers import ALICE, LifecyclePlatform, is_message_post, page


def message_content(request: httpx.Request) -> str:
    return json.loads(request.content)["message"]["content"]


class ActivityPlatform(LifecyclePlatform):
    """A running room whose participants are listed, whose messages land, and
    whose activity and event posts are recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.room_messages: list[str] = []
        self.activity_bodies: list[dict[str, Any]] = []
        self.hold_next_activity = False
        self.activity_entered = asyncio.Event()
        self.activity_release = asyncio.Event()
        self.activity_cancelled = asyncio.Event()
        self.reject_detail = False
        self.event_bodies: list[dict[str, Any]] = []

    @property
    def working_transitions(self) -> list[bool]:
        return [
            working
            for working, _ in groupby(body["working"] for body in self.activity_bodies)
        ]

    async def respond(self, request: httpx.Request) -> httpx.Response:
        response = self.answer(request)
        if self.hold_next_activity and request.url.path.endswith("/activity"):
            self.hold_next_activity = False
            self.activity_entered.set()
            try:
                await self.activity_release.wait()
            except asyncio.CancelledError:
                self.activity_cancelled.set()
                raise
        return response

    def assert_hides(self, *private: str) -> None:
        serialized = json.dumps(self.activity_bodies)
        for value in private:
            assert value not in serialized

    def assert_no_tool_narration(self) -> None:
        assert not any(
            body.get("message_type") in {"tool_call", "tool_result"}
            for body in self.event_bodies
        )

    def answer(self, request: httpx.Request) -> httpx.Response:
        tail = request.url.path.rsplit("/", 1)[-1]
        if tail == "events" and request.method == "POST":
            self.event_bodies.append(json.loads(request.content))
            if not self.stopped:
                # A running room accepts narration; falling through to the
                # lifecycle base would answer 403, which the platform client
                # reads as a stopped execution and tears the turn machinery
                # down around the first reported tool call.
                return httpx.Response(
                    201,
                    json={
                        "data": {
                            "id": "e-1",
                            "message_type": "event",
                            "success": True,
                        }
                    },
                )
        if tail == "activity":
            body = json.loads(request.content)
            self.activity_bodies.append(body)
            if self.reject_detail and "activity" in body:
                return httpx.Response(422, json={"error": "unsupported activity"})
        if tail == "participants":
            return page([{**ALICE, "role": "member", "status": "active"}])
        if is_message_post(request):
            self.room_messages.append(message_content(request))
            return httpx.Response(
                201, json={"data": {"id": "m-1", "success": True, "recipients": []}}
            )
        return super().answer(request)


class IdleReportGate:
    """Holds the room's ``working: false`` report while ``release`` is clear."""

    def __init__(self, peer: ActivityPlatform) -> None:
        self.peer = peer
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def answer(self, request: httpx.Request) -> httpx.Response:
        response = await self.peer.respond(request)
        if (
            request.url.path.endswith("/activity")
            and not json.loads(request.content)["working"]
        ):
            self.entered.set()
            await self.release.wait()
        return response


class HeldWorkerHandler:
    """A message handler whose every turn binds one worker held on ``release``."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.workers: list[TurnTask] = []

    async def __call__(self, ctx: ExecutionContext, event: PlatformEvent) -> None:
        worker = asyncio.create_task(self.release.wait())
        self.workers.append(bind_turn_task(AgentTools.from_context(ctx), worker))

    async def finish(self, ctx: ExecutionContext) -> None:
        """Release every worker and join the activity report their end starts."""
        self.release.set()
        await asyncio.gather(*(worker.task for worker in self.workers))
        await asyncio.sleep(0)
        await ctx._join_activity_completion()


async def consume_control(ctx: ExecutionContext, event: PlatformEvent) -> bool:
    return True


class HeldReplyPlatform(ActivityPlatform):
    """Holds the post of one reply until ``release`` is set, so its turn can be
    kept in the delivery tail after the model has finished."""

    def __init__(self, held_reply: str) -> None:
        super().__init__()
        self.held_reply = held_reply
        self.reply_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def respond(self, request: httpx.Request) -> httpx.Response:
        if is_message_post(request) and message_content(request) == self.held_reply:
            self.reply_entered.set()
            await self.release.wait()
        return await super().respond(request)


class ObservedIdle(asyncio.Event):
    """The idle event a graceful stop waits on, announcing that wait."""

    def __init__(self) -> None:
        super().__init__()
        self.set()
        self.awaited = asyncio.Event()

    async def wait(self) -> Literal[True]:
        self.awaited.set()
        return await super().wait()
