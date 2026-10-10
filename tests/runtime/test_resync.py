"""Tests for idle-timeout resync and reconnect resync (INT-333).

Covers:
- request_resync() enqueues ResyncRequest sentinel
- Sentinel wakes Phase 2 loop and calls _resync_pending_messages()
- Idle timeout calls _resync_pending_messages() within the configured interval
- Idle polls are jittered and back off while they find nothing to run
- _resync_pending_messages() happy path: processes missed message
- _resync_pending_messages() empty path: /next returns None, no error
- AgentRuntime._on_reconnected() calls request_resync() on all executions
- AgentRuntime._on_reconnected() skips executions without request_resync (custom impls)
- RoomPresence._handle_reconnect() fires on_reconnected even when auto_subscribe_existing=False
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from band.runtime.execution import ExecutionContext, ResyncRequest
from band.runtime.presence import RoomPresence
from band.runtime.resync_backoff import IdleResyncBackoff
from band.runtime.runtime import AgentRuntime
from band.runtime.types import PlatformMessage, SessionConfig
from tests.conftest import make_message_event
from tests.runtime.conftest import admit_room, wait_for_condition

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_mock_link():
    """BandLink mock configured for ExecutionContext tests."""
    link = MagicMock()
    link.agent_id = "agent-123"
    link.is_connected = False

    link.connect = AsyncMock()
    link.subscribe_agent_rooms = AsyncMock()
    link.subscribe_room = AsyncMock()
    link.unsubscribe_room = AsyncMock()

    link.rest = MagicMock()
    link.rest.agent_api_participants = MagicMock()
    link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=MagicMock(data=[])
    )
    link.rest.agent_api_context = MagicMock()
    link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
        return_value=MagicMock(data=[])
    )
    link.rest.agent_api_chats = MagicMock()
    api_response = MagicMock()
    api_response.data = []
    api_response.metadata = MagicMock()
    api_response.metadata.total_pages = None
    link.rest.agent_api_chats.list_agent_chats = AsyncMock(return_value=api_response)

    link.mark_processing = AsyncMock()
    link.mark_processed = AsyncMock()
    link.mark_failed = AsyncMock()
    link.get_next_message = AsyncMock(return_value=None)
    link.get_stale_processing_messages = AsyncMock(return_value=[])
    link.report_activity = AsyncMock(return_value=True)

    async def empty_aiter():
        return
        yield

    link.__aiter__ = lambda self: empty_aiter()

    return link


@pytest.fixture
def mock_link():
    return make_mock_link()


@pytest.fixture
def mock_handler():
    return AsyncMock()


def make_platform_message(
    msg_id: str = "msg-1", room_id: str = "room-1"
) -> PlatformMessage:
    return PlatformMessage(
        id=msg_id,
        room_id=room_id,
        content="Hello",
        sender_id="user-999",
        sender_type="User",
        sender_name="Tester",
        message_type="text",
        metadata={},
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# TestRequestResync
# ---------------------------------------------------------------------------


class TestRequestResync:
    """Tests for ExecutionContext.request_resync()."""

    async def test_enqueues_resync_sentinel(self, mock_link, mock_handler):
        """request_resync() should put a ResyncRequest onto the queue."""
        ctx = ExecutionContext("room-1", mock_link, mock_handler)

        await ctx.request_resync()

        assert ctx.queue.qsize() == 1
        item = ctx.queue.get_nowait()
        assert isinstance(item, ResyncRequest)

    async def test_sentinel_triggers_resync(self, mock_link, mock_handler):
        """Enqueueing a sentinel should cause the Phase 2 loop to call /next."""
        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        await ctx.start()

        await wait_for_condition(lambda: ctx._sync_complete)

        call_count_before = mock_link.get_next_message.call_count

        await ctx.request_resync()
        await wait_for_condition(
            lambda: mock_link.get_next_message.call_count > call_count_before
        )

        # At least one more /next call should have occurred
        assert mock_link.get_next_message.call_count > call_count_before

        await ctx.stop()

    async def test_multiple_resyncs_dont_crash(self, mock_link, mock_handler):
        """Multiple rapid request_resync() calls should not crash or deadlock."""
        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        await ctx.start()
        await wait_for_condition(lambda: ctx._sync_complete)

        for _ in range(5):
            await ctx.request_resync()

        await wait_for_condition(lambda: ctx.queue.qsize() == 0)
        # Still running, no exception
        assert ctx.is_running

        await ctx.stop()


# ---------------------------------------------------------------------------
# TestIdleTimeout
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    """Tests for idle-timeout resync in Phase 2 loop."""

    async def test_idle_timeout_triggers_resync(self, mock_link, mock_handler):
        """Phase 2 should call /next within idle_resync_seconds with no WS events."""
        config = SessionConfig(idle_resync_seconds=0.01)  # fast timeout for test
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        await ctx.start()

        await wait_for_condition(
            lambda: mock_link.get_next_message.call_count >= 2,
            timeout=1.0,
        )

        # get_next_message is called during Phase 1 AND during idle timeout resync
        assert mock_link.get_next_message.call_count >= 2

        await ctx.stop()

    async def test_idle_timeout_does_not_fire_when_events_arrive(
        self, mock_link, mock_handler
    ):
        """If events arrive before timeout, resync should not add extra /next calls."""
        config = SessionConfig(idle_resync_seconds=60)  # very long timeout
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        await ctx.start()
        await wait_for_condition(lambda: ctx._sync_complete)

        call_count_after_phase1 = mock_link.get_next_message.call_count

        # Send a real WS event — this resets the idle timer
        event = make_message_event(room_id="room-1", msg_id="msg-x")
        await ctx.on_event(event)
        await wait_for_condition(lambda: mock_handler.await_count >= 1)

        # No idle timeout should have fired (timeout is 60s)
        assert mock_link.get_next_message.call_count == call_count_after_phase1

        await ctx.stop()


# ---------------------------------------------------------------------------
# TestIdleResyncBackoff
# ---------------------------------------------------------------------------


class TestIdleResyncBackoffPolicy:
    """The idle /next interval: jittered, doubling while polls find nothing."""

    def test_wait_is_jittered_within_the_upper_half_of_the_interval(self):
        low = IdleResyncBackoff(60, 300, random=lambda: 0.0)
        high = IdleResyncBackoff(60, 300, random=lambda: 1.0)

        assert low.next_wait() == 30
        assert high.next_wait() == 60

    def test_empty_polls_double_the_interval_up_to_the_cap(self):
        backoff = IdleResyncBackoff(60, 300, random=lambda: 1.0)

        waits = []
        for _ in range(5):
            waits.append(backoff.next_wait())
            backoff.found_nothing()

        assert waits == [60, 120, 240, 300, 300]

    def test_reset_returns_to_the_base_interval(self):
        backoff = IdleResyncBackoff(60, 300, random=lambda: 1.0)
        backoff.found_nothing()
        backoff.found_nothing()

        backoff.reset()

        assert backoff.next_wait() == 60

    def test_cap_below_the_base_interval_keeps_the_base(self):
        backoff = IdleResyncBackoff(600, 300, random=lambda: 1.0)
        backoff.found_nothing()

        assert backoff.next_wait() == 600


class TestIdleResyncBackoffLoop:
    """Phase 2 applies the backoff to its idle /next polls."""

    async def test_empty_idle_polls_back_off_to_the_cap(self, mock_link, mock_handler):
        config = SessionConfig(idle_resync_seconds=0.01, idle_resync_max_seconds=0.04)
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        await ctx.start()

        # Phase 1 sync, then the 0.01 -> 0.02 -> 0.04 polls.
        await wait_for_condition(
            lambda: mock_link.get_next_message.call_count >= 5, timeout=2.0
        )

        assert ctx._idle_resync.level == 0.04
        await ctx.stop()

    async def test_backed_off_room_polls_less_often(self, mock_link, mock_handler):
        # Without backoff 0.05 s polls make at least ~20 calls in a second;
        # doubling to 0.4 s leaves well under ten.
        config = SessionConfig(idle_resync_seconds=0.05, idle_resync_max_seconds=0.4)
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        await ctx.start()

        await asyncio.sleep(1.0)

        assert mock_link.get_next_message.call_count < 10
        await ctx.stop()

    async def test_idle_waits_use_the_jittered_draw(self, mock_handler, monkeypatch):
        # Record the deadline the room's own loop arms while it waits for a
        # push: it must be the jittered draw, not the bare interval.
        real_timeout = asyncio.timeout
        waits = {}
        for draw in (0.0, 1.0):
            config = SessionConfig(idle_resync_seconds=0.1, idle_resync_max_seconds=0.1)
            ctx = ExecutionContext(
                "room-1", make_mock_link(), mock_handler, config=config
            )
            ctx._idle_resync = IdleResyncBackoff(0.1, 0.1, random=lambda d=draw: d)
            armed = asyncio.Queue()

            def recording_timeout(delay, ctx=ctx, armed=armed):
                if asyncio.current_task() is ctx._process_loop_task:
                    armed.put_nowait(delay)
                return real_timeout(delay)

            monkeypatch.setattr(asyncio, "timeout", recording_timeout)
            await ctx.start()
            waits[draw] = await asyncio.wait_for(armed.get(), timeout=2.0)
            await ctx.stop()

        assert waits == {0.0: 0.05, 1.0: 0.1}

    async def test_websocket_message_resets_the_interval(self, mock_link, mock_handler):
        config = SessionConfig(idle_resync_seconds=0.01, idle_resync_max_seconds=0.04)
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        # Read when the pushed message is handled: later idle polls in this
        # quiet room grow the interval again.
        levels = []
        mock_handler.side_effect = lambda *_: levels.append(ctx._idle_resync.level)
        await ctx.start()
        await wait_for_condition(lambda: ctx._idle_resync.level == 0.04, timeout=2.0)

        await ctx.on_event(make_message_event(room_id="room-1", msg_id="msg-ws"))
        await wait_for_condition(lambda: mock_handler.await_count >= 1)

        assert levels == [0.01]
        await ctx.stop()

    async def test_poll_that_finds_a_missed_message_keeps_the_base_interval(
        self, mock_link, mock_handler
    ):
        calls = 0

        async def next_message(room_id):
            # Phase 1 finds nothing; afterwards every idle poll finds one
            # missed message before the backlog is empty again.
            nonlocal calls
            calls += 1
            if calls > 1 and calls % 2 == 0:
                return make_platform_message(msg_id=f"missed-{calls}", room_id=room_id)
            return None

        mock_link.get_next_message.side_effect = next_message
        config = SessionConfig(idle_resync_seconds=0.01, idle_resync_max_seconds=0.04)
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        # Record the interval each wait is drawn from, which the loop decides
        # only after the whole poll has finished.
        levels = []
        backoff = ctx._idle_resync
        draw = backoff.next_wait

        def recording_next_wait():
            levels.append(backoff.level)
            return draw()

        backoff.next_wait = recording_next_wait
        await ctx.start()

        await wait_for_condition(lambda: len(levels) >= 4, timeout=2.0)

        assert levels[:4] == [0.01] * 4
        await ctx.stop()

    async def test_poll_offered_only_a_message_it_cannot_run_still_backs_off(
        self, mock_link, mock_handler
    ):
        # A head the agent will never run is not traffic: the room is as
        # quiet as one whose /next is empty.
        stuck = make_platform_message(msg_id="msg-failed", room_id="room-1")
        mock_link.get_next_message.return_value = stuck
        config = SessionConfig(idle_resync_seconds=0.01, idle_resync_max_seconds=0.04)
        ctx = ExecutionContext("room-1", mock_link, mock_handler, config=config)
        ctx._retry_tracker.mark_permanently_failed("msg-failed")
        await ctx.start()

        await wait_for_condition(lambda: ctx._idle_resync.level == 0.04, timeout=2.0)

        mock_handler.assert_not_called()
        await ctx.stop()

    def test_rejects_a_non_positive_cap(self):
        with pytest.raises(ValueError, match="idle_resync_max_seconds"):
            SessionConfig(idle_resync_max_seconds=0)


# ---------------------------------------------------------------------------
# TestResyncPendingMessages
# ---------------------------------------------------------------------------


class TestResyncPendingMessages:
    """Tests for ExecutionContext._resync_pending_messages()."""

    async def test_empty_returns_immediately(self, mock_link, mock_handler):
        """/next returning None immediately should not call the handler."""
        mock_link.get_next_message.return_value = None

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        # Call directly without starting the loop
        await ctx._resync_pending_messages()

        mock_handler.assert_not_called()

    async def test_processes_single_missed_message(self, mock_link, mock_handler):
        """/next returning one message then None should process that message."""
        msg = make_platform_message(msg_id="missed-1", room_id="room-1")
        # First call returns the missed message, second returns None
        mock_link.get_next_message.side_effect = [msg, None]

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        await ctx.start()
        await wait_for_condition(lambda: ctx._sync_complete)

        # Reset state so _resync_pending_messages runs fresh
        mock_link.get_next_message.side_effect = [msg, None]
        await ctx._resync_pending_messages()

        # Handler should have been called for the missed message
        mock_handler.assert_called()

        await ctx.stop()

    async def test_skips_duplicate_message(self, mock_link, mock_handler):
        """A message already recorded as completed should be skipped."""
        msg = make_platform_message(msg_id="dup-1", room_id="room-1")
        mock_link.get_next_message.side_effect = [msg, None]

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        # Mark the message as already processed
        ctx.claims.remember_completed(ctx.room_id, "dup-1")

        await ctx._resync_pending_messages()

        mock_handler.assert_not_called()

    async def test_exception_does_not_propagate(self, mock_link, mock_handler):
        """Exceptions inside _resync_pending_messages should be caught, not raised."""
        mock_link.get_next_message.side_effect = RuntimeError("API down")

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        # Should complete without raising
        await ctx._resync_pending_messages()


# ---------------------------------------------------------------------------
# TestDrainProgressGuard
# ---------------------------------------------------------------------------


class RepeatingNextMessage:
    """A ``/next`` stub that keeps returning the same message — what the
    endpoint does while that message stays the oldest not-processed one."""

    def __init__(self, msg: PlatformMessage) -> None:
        self.msg = msg
        self.calls = 0

    async def __call__(self, room_id: str) -> PlatformMessage:
        self.calls += 1
        if self.calls > 10:
            raise RuntimeError("/next drain did not terminate")
        return self.msg


class TestDrainProgressGuard:
    """Tests for /next drains refusing to spin on a message they cannot advance."""

    async def test_resync_ends_when_next_repeats_a_skipped_message(
        self, mock_link, mock_handler
    ):
        msg = make_platform_message(msg_id="wedged-1", room_id="room-1")
        next_message = RepeatingNextMessage(msg)
        mock_link.get_next_message.side_effect = next_message.__call__

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        ctx.claims.remember_completed(ctx.room_id, "wedged-1")

        assert await ctx._resync_pending_messages() is True
        assert next_message.calls == 2

    async def test_startup_sync_ends_when_next_repeats_a_skipped_message(
        self, mock_link, mock_handler
    ):
        msg = make_platform_message(msg_id="wedged-1", room_id="room-1")
        next_message = RepeatingNextMessage(msg)
        mock_link.get_next_message.side_effect = next_message.__call__

        ctx = ExecutionContext("room-1", mock_link, mock_handler)
        ctx.claims.remember_completed(ctx.room_id, "wedged-1")

        assert await ctx._synchronize_with_next() is True
        assert next_message.calls == 2

    async def test_startup_sync_retries_failed_message_in_same_drain(self, mock_link):
        # A turn that fails with retry budget left stays actionable, so /next
        # offers it again in the same drain and the retry must run there.
        calls = []

        async def handler(_ctx: object, _event: object) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient turn failure")

        msg = make_platform_message(msg_id="retry-1", room_id="room-1")
        mock_link.get_next_message.side_effect = [msg, msg, None]

        ctx = ExecutionContext(
            "room-1", mock_link, handler, config=SessionConfig(max_message_retries=2)
        )

        assert await ctx._synchronize_with_next() is True
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# TestAgentRuntimeOnReconnected
# ---------------------------------------------------------------------------


class TestAgentRuntimeOnReconnected:
    """Tests for AgentRuntime._on_reconnected()."""

    async def test_calls_request_resync_on_all_executions(
        self, mock_link, mock_handler
    ):
        """_on_reconnected() should call request_resync() on each execution."""
        runtime = AgentRuntime(mock_link, "agent-123", mock_handler)

        exec1 = MagicMock()
        exec1.request_resync = AsyncMock()
        exec2 = MagicMock()
        exec2.request_resync = AsyncMock()

        runtime.executions = {"room-1": exec1, "room-2": exec2}

        await runtime._on_reconnected()

        exec1.request_resync.assert_called_once()
        exec2.request_resync.assert_called_once()

    async def test_skips_execution_without_request_resync(
        self, mock_link, mock_handler
    ):
        """_on_reconnected() should not raise if an execution lacks request_resync."""
        runtime = AgentRuntime(mock_link, "agent-123", mock_handler)

        # Simulate a legacy/custom Execution without the new method
        legacy_exec = MagicMock(spec=[])  # spec=[] → no attributes at all

        runtime.executions = {"room-legacy": legacy_exec}

        # Should not raise AttributeError
        await runtime._on_reconnected()

    async def test_one_failure_does_not_abort_others(self, mock_link, mock_handler):
        """A failing request_resync() should not prevent the others from running."""
        runtime = AgentRuntime(mock_link, "agent-123", mock_handler)

        exec1 = MagicMock()
        exec1.request_resync = AsyncMock(side_effect=RuntimeError("boom"))
        exec2 = MagicMock()
        exec2.request_resync = AsyncMock()

        runtime.executions = {"room-1": exec1, "room-2": exec2}

        await runtime._on_reconnected()

        exec2.request_resync.assert_called_once()

    async def test_presence_reconnect_still_resyncs_active_executions_if_api_fails(
        self, mock_link, mock_handler
    ):
        """The full presence -> runtime callback chain should survive API failure."""
        runtime = AgentRuntime(mock_link, "agent-123", mock_handler)

        execution = MagicMock()
        execution.request_resync = AsyncMock()
        runtime.executions = {"room-1": execution}

        mock_link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        await runtime.presence._handle_reconnect()

        execution.request_resync.assert_called_once()


# ---------------------------------------------------------------------------
# TestPresenceReconnectOnReconnectedCallback
# ---------------------------------------------------------------------------


class TestPresenceReconnectOnReconnectedCallback:
    """Tests that on_reconnected fires reliably from _handle_reconnect."""

    @pytest.fixture
    def mock_presence_link(self):
        link = MagicMock()
        link.agent_id = "agent-123"
        link.is_connected = False
        link.connect = AsyncMock()
        link.subscribe_agent_rooms = AsyncMock()
        link.subscribe_room = AsyncMock()
        link.unsubscribe_room = AsyncMock()

        link.rest = MagicMock()
        link.rest.agent_api_chats = MagicMock()

        # Return a properly structured response so _list_existing_rooms terminates
        # correctly (total_pages=None breaks the pagination loop after one call).
        api_response = MagicMock()
        api_response.data = []
        api_response.metadata = MagicMock()
        api_response.metadata.total_pages = None
        link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=api_response
        )

        async def empty_aiter():
            return
            yield

        link.__aiter__ = lambda self: empty_aiter()
        return link

    async def test_on_reconnected_fires_with_auto_subscribe_true(
        self, mock_presence_link
    ):
        """on_reconnected fires after reconnect when auto_subscribe_existing=True."""
        reconnected_calls = []

        async def on_reconnected():
            reconnected_calls.append(1)

        presence = RoomPresence(mock_presence_link, auto_subscribe_existing=True)
        presence.on_reconnected = on_reconnected

        await presence._handle_reconnect()

        assert len(reconnected_calls) == 1

    async def test_on_reconnected_fires_with_auto_subscribe_false(
        self, mock_presence_link
    ):
        """on_reconnected fires after reconnect even when auto_subscribe_existing=False.

        The finally block ensures the callback always fires regardless of early
        returns in the try block (which exits early when auto_subscribe_existing=False).
        """
        reconnected_calls = []

        async def on_reconnected():
            reconnected_calls.append(1)

        presence = RoomPresence(mock_presence_link, auto_subscribe_existing=False)
        presence.on_reconnected = on_reconnected

        await presence._handle_reconnect()

        assert len(reconnected_calls) == 1

    async def test_on_reconnected_still_fires_if_api_fails(self, mock_presence_link):
        """on_reconnected should still fire if room reconciliation fails.

        Existing executions still need a /next resync even when the room-list API
        is temporarily unavailable during reconnect.
        """
        reconnected_calls = []

        async def on_reconnected():
            reconnected_calls.append(1)

        mock_presence_link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        presence = RoomPresence(mock_presence_link, auto_subscribe_existing=True)
        presence.on_reconnected = on_reconnected

        await presence._handle_reconnect()

        assert len(reconnected_calls) == 1

    async def test_on_reconnected_still_fires_if_unsubscribe_fails(
        self, mock_presence_link
    ):
        """on_reconnected should still fire if a stale room unsubscribe fails."""
        reconnected_calls = []

        async def on_reconnected():
            reconnected_calls.append(1)

        room = MagicMock()
        room.id = "room-new"
        room.model_dump.return_value = {"id": "room-new"}

        api_response = MagicMock()
        api_response.data = [room]
        api_response.metadata = MagicMock()
        api_response.metadata.total_pages = None
        mock_presence_link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=api_response
        )
        mock_presence_link.unsubscribe_room = AsyncMock(
            side_effect=RuntimeError("unsubscribe failed")
        )

        presence = RoomPresence(mock_presence_link, auto_subscribe_existing=True)
        admit_room(presence, "room-old")
        presence.on_reconnected = on_reconnected

        await presence._handle_reconnect()

        assert len(reconnected_calls) == 1

    async def test_cancelled_error_propagates_from_callback(self, mock_presence_link):
        """CancelledError raised in on_reconnected must propagate (structured concurrency)."""

        async def on_reconnected_that_raises():
            raise asyncio.CancelledError

        presence = RoomPresence(mock_presence_link, auto_subscribe_existing=False)
        presence.on_reconnected = on_reconnected_that_raises

        with pytest.raises(asyncio.CancelledError):
            await presence._handle_reconnect()
