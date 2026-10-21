from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from band.core.activity import sync_activity_tool_names
from band.core.types import Emit
from band.platform.event import PlatformEvent
from band.runtime.execution import ExecutionContext
from band.runtime.tools.agent import AgentTools
from band.runtime.working_state import (
    report_tool_activity,
    tool_call_activity,
)
from tests.conftest import make_message_event
from tests.runtime.activity.kit import (
    ActivityPeer,
    DetailPeer,
    activity_context,
    reporting_handler,
)


class TestActivityToolNameGate:
    async def test_tool_names_report_while_tool_calls_are_emitted(self) -> None:
        peer = DetailPeer()
        async with activity_context(
            peer,
            reporting_handler(tool="Bash"),
            AsyncMock(),
        ) as ctx:
            await ctx._process_event(make_message_event(msg_id="trigger"))
        kinds = [detail and detail.get("kind") for detail in peer.details]
        assert kinds.count("tool_call") == 1
        named = [d for d in peer.details if d and d.get("kind") == "tool_call"]
        assert named[0]["name"] == "Bash"

    async def test_tool_names_are_dropped_when_tool_calls_are_not_emitted(
        self,
    ) -> None:
        peer = DetailPeer()
        async with activity_context(
            peer,
            reporting_handler(tool="Bash"),
            AsyncMock(),
        ) as ctx:
            ctx.set_activity_tool_names(False)
            await ctx._process_event(make_message_event(msg_id="trigger"))
        kinds = [detail and detail.get("kind") for detail in peer.details]
        assert "tool_call" not in kinds
        assert "reasoning" in kinds
        for detail in peer.details:
            assert detail is None or not detail.get("name")

    async def test_permission_and_memory_survive_the_name_gate(self) -> None:
        peer = DetailPeer()
        async with activity_context(
            peer,
            reporting_handler(permission=True, memory=True),
            AsyncMock(),
        ) as ctx:
            ctx.set_activity_tool_names(False)
            await ctx._process_event(make_message_event(msg_id="trigger"))
        kinds = [detail and detail.get("kind") for detail in peer.details]
        assert "memory" in kinds
        assert "permission" in kinds

    async def test_flip_after_binding_hides_names_in_the_running_turn(
        self,
    ) -> None:
        # The codex ordering: the observer is bound to the turn before the
        # adapter's emit sync flips the flag. The gate must read the live
        # choice at report time, not the choice at construction time.
        peer = DetailPeer()
        bound: list = []

        async def handler(ctx: ExecutionContext, event: PlatformEvent) -> None:
            bound.append(ctx.activity_observer(ctx.current_scope))
            ctx.set_activity_tool_names(False)
            await bound[0].update(tool_call_activity("Bash"))

        async with activity_context(peer, handler, AsyncMock()) as ctx:
            await ctx._process_event(make_message_event(msg_id="trigger"))
        tool_frames = [d for d in peer.details if d and d.get("kind") == "tool_call"]
        assert not tool_frames
        assert {"kind": "reasoning"} in peer.details

    async def test_sync_resolves_names_from_the_emit_choice(self) -> None:
        async with activity_context(
            ActivityPeer(), reporting_handler(), AsyncMock()
        ) as ctx:
            tools = SimpleNamespace(_ctx=ctx)
            sync_activity_tool_names(tools, frozenset({Emit.TOOL_CALLS}))
            assert ctx._activity_tool_names is True
            sync_activity_tool_names(tools, frozenset({Emit.THOUGHTS}))
            assert ctx._activity_tool_names is False
            sync_activity_tool_names(SimpleNamespace(), frozenset())
            assert ctx._activity_tool_names is False

    async def test_continuation_frame_after_takeover_never_reaches_wire(
        self,
    ) -> None:
        """An observer that outlives its turn (a background continuation) is
        fenced once the next room-message turn owns activity: dropping the
        ownership check in ``observe_activity`` surfaces a third named frame."""
        peer = DetailPeer()
        bound: list = []
        turns = {"n": 0}

        async def handler(ctx: ExecutionContext, event: PlatformEvent) -> None:
            turns["n"] += 1
            tools = AgentTools.from_context(ctx)
            if turns["n"] == 1:
                bound.append(ctx.activity_observer(ctx.current_scope))
                await report_tool_activity(tools, "bash")
                return
            # The new turn owns activity; the continuation then wakes up.
            await report_tool_activity(tools, "exec")
            await bound[0].update(tool_call_activity("band_send_message"))

        async with activity_context(peer, handler, AsyncMock()) as ctx:
            await ctx._process_event(make_message_event(msg_id="turn-one"))
            await ctx._process_event(make_message_event(msg_id="turn-two"))
        named = [
            d.get("name") for d in peer.details if d and d.get("kind") == "tool_call"
        ]
        assert named == ["bash", "exec"]
