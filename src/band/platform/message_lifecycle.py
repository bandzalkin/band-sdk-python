"""Message-lifecycle REST operations — no WebSocket state involved.

Split out of BandLink: mark_processing/processed/failed, report_activity,
and the /next + paginated backlog REST reads share nothing with WebSocket
connection or subscription state, only a REST client.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any

from band_rest.core.api_error import ApiError
from band_rest.types.chat_message import ChatMessage
from band_rest.types.chat_message_metadata import ChatMessageMetadata

from band.client.rest import DEFAULT_REQUEST_OPTIONS, AsyncRestClient
from band.core.exceptions import RoomExecutionStoppedError
from band.core.types import metadata_to_dict
from band.runtime.types import PlatformMessage

logger = logging.getLogger(__name__)


def _message_metadata(metadata: ChatMessageMetadata | None) -> dict[str, object]:
    """Normalize a Fern-typed message metadata into the plain dict PlatformMessage carries."""
    return metadata_to_dict(metadata, exclude_none=True)


def _platform_message(item: ChatMessage, room_id: str) -> PlatformMessage:
    return PlatformMessage(
        id=item.id,
        room_id=item.chat_room_id or room_id,
        content=item.content,
        sender_id=item.sender_id,
        sender_type=item.sender_type,
        sender_name=item.sender_name or "",
        message_type=item.message_type,
        metadata=_message_metadata(item.metadata),
        created_at=item.inserted_at or datetime.now(UTC),
    )


class MessageLifecycle:
    """Message mark/report/fetch operations for one agent's REST client.

    ``rest`` is taken per call, not cached at construction: the caller
    (``BandLink``) owns the REST client and may swap it at any point, so
    every call here uses whatever ``rest`` the caller currently has rather
    than a snapshot from construction time.
    """

    def __init__(self) -> None:
        # Debounces activity-report warnings: keep-alive runs every few
        # seconds, so a down endpoint would otherwise flood the log.
        self._activity_report_failing = False

    async def mark_processing(
        self, rest: AsyncRestClient, room_id: str, message_id: str
    ) -> bool:
        """
        Mark message as being processed on the server.

        This does NOT remove it from /next: the actionable set excludes only
        'processed', so a crashed or stopped attempt stays replayable. Only
        mark_processed clears the message from /next.

        Returns False when the call fails; the caller must not run the turn.

        Raises:
            RoomExecutionStoppedError: When the platform refuses the claim
                because this room's agent execution is stopped. The message
                stays actionable and replays via /next once the execution is
                resumed (a play signal).
        """
        logger.debug("Marking message %s as processing", message_id)
        try:
            # The raw response is the only place the status code survives:
            # the plain client returns ``None`` for any empty body, so it
            # cannot tell the stopped room's 204 from a malformed reply.
            response = await rest.agent_api_messages.with_raw_response.mark_agent_message_processing(
                chat_id=room_id,
                id=message_id,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
        except Exception as e:  # noqa: BLE001 -- best-effort event emission must not crash the turn/link
            logger.warning("Failed to mark message %s as processing: %s", message_id, e)
            return False
        # A claim is answered 200 with the updated message; the platform
        # answers 204 only when it refuses the claim because this room's
        # execution is stopped. Reading that as success would run a turn whose
        # every post is then rejected.
        if response.status_code == HTTPStatus.NO_CONTENT:
            raise RoomExecutionStoppedError(room_id)
        return True

    async def mark_processed(
        self, rest: AsyncRestClient, room_id: str, message_id: str
    ) -> bool:
        """
        Mark message as successfully processed on the server.

        Clears the message from unprocessed queue.
        """
        logger.debug("Marking message %s as processed", message_id)
        try:
            await rest.agent_api_messages.mark_agent_message_processed(
                chat_id=room_id,
                id=message_id,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
        except Exception as e:  # noqa: BLE001 -- best-effort event emission must not crash the turn/link
            logger.warning("Failed to mark message %s as processed: %s", message_id, e)
            return False
        return True

    async def mark_failed(
        self, rest: AsyncRestClient, room_id: str, message_id: str, error: str
    ) -> bool:
        """
        Mark message as failed on the server.

        Records the error and may trigger retry logic on the server side.
        """
        error = error.strip() or "Unknown error"
        logger.warning("Marking message %s as failed: %s", message_id, error)
        try:
            await rest.agent_api_messages.mark_agent_message_failed(
                chat_id=room_id,
                id=message_id,
                error=error,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
        except Exception as e:  # noqa: BLE001 -- best-effort event emission must not crash the turn/link
            logger.warning("Failed to mark message %s as failed: %s", message_id, e)
            return False
        return True

    async def report_activity(
        self,
        rest: AsyncRestClient,
        room_id: str,
        working: bool,
        *,
        timeout_seconds: int = 2,
    ) -> bool:
        """
        Report the agent's boolean working state for a room's execution.

        ``working=True`` while a reasoning cycle is active (refreshed on a
        keep-alive cadence), ``False`` once it ends. Never raises — failures
        are swallowed and returned as ``False``, since the platform's TTL is
        the backstop and activity reporting must never break message
        processing.

        ``timeout_seconds`` bounds each POST so a slow/half-open endpoint
        can't stall the keep-alive or wedge teardown. Retries are off: a
        dropped keep-alive gets re-sent next cadence tick, and a dropped
        ``false`` is cleared by the platform TTL either way — retrying would
        only add latency.
        """
        try:
            await rest.agent_api_activity.report_agent_chat_activity(
                chat_id=room_id,
                working=working,
                request_options={
                    "timeout_in_seconds": timeout_seconds,
                    "max_retries": 0,
                },
            )
        except Exception as e:  # noqa: BLE001 -- best-effort event emission must not crash the turn/link
            if not self._activity_report_failing:
                self._activity_report_failing = True
                logger.warning(
                    "Failed to report activity (working=%s) for room %s: %s; "
                    "suppressing repeat warnings until recovery",
                    working,
                    room_id,
                    e,
                )
            else:
                logger.debug(
                    "Activity report still failing (working=%s) for room %s: %s",
                    working,
                    room_id,
                    e,
                )
            return False
        if self._activity_report_failing:
            self._activity_report_failing = False
            logger.info("Activity reporting recovered for room %s", room_id)
        return True

    async def get_next_message(
        self, rest: AsyncRestClient, room_id: str
    ) -> PlatformMessage | None:
        """
        Get the next actionable message for a room from the server.

        Returns ``None`` only when the platform reports 204 (nothing
        pending) — never to mean "the call failed."

        Raises:
            ApiError: non-204 REST failure.
            Exception: transport-level failure (connection error, timeout).

        Callers that want to swallow transient failures must wrap this call
        themselves: conflating "no pending" with "lookup failed" used to
        silently drop messages at the claim step.
        """
        logger.debug("Getting next message for room %s", room_id)
        try:
            response = await rest.agent_api_messages.get_agent_next_message(
                chat_id=room_id,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
        except ApiError as e:
            # 204 No Content means no actionable messages — the only "None"
            # case the platform expresses through an ApiError.
            if e.status_code == 204:
                logger.debug("No actionable messages for room %s", room_id)
                return None
            logger.warning("Failed to get next message: %s", e)
            raise

        if response is None or response.data is None:
            return None

        return _platform_message(response.data, room_id)

    async def get_stale_processing_messages(
        self, rest: AsyncRestClient, room_id: str
    ) -> list[PlatformMessage]:
        """
        Get messages stuck in 'processing' state for a room.

        Listing errors propagate so the runtime can retry startup instead of
        treating an unavailable recovery sweep as an empty room.
        """
        return await self._list_messages(rest, room_id, status="processing")

    async def get_actionable_messages(
        self, rest: AsyncRestClient, room_id: str
    ) -> list[PlatformMessage]:
        """Snapshot actionable deliveries in authoritative server order.

        Finish every page before the runtime changes delivery statuses: page
        offsets shift when messages leave the filtered set. A single unfiltered
        listing also keeps status transitions from hiding older deliveries.
        """
        messages: dict[str, PlatformMessage] = {}
        for message in await self._list_messages(rest, room_id, status=None):
            messages.setdefault(message.id, message)
        return list(messages.values())

    async def _list_messages(
        self,
        rest: AsyncRestClient,
        room_id: str,
        *,
        status: str | None,
    ) -> list[PlatformMessage]:
        messages: list[PlatformMessage] = []
        page = 1
        pagination: dict[str, Any] = {"sort_order": "asc"}
        seen_cursors: set[str] = set()
        while True:
            response = await rest.agent_api_messages.list_agent_messages(
                chat_id=room_id,
                status=status,
                **pagination,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            messages.extend(_platform_message(item, room_id) for item in response.data)

            cursor = response.metadata.next_cursor
            if response.metadata.has_more is True and cursor:
                if cursor in seen_cursors:
                    raise ValueError("Message listing has no usable pagination cursor")
                seen_cursors.add(cursor)
                pagination = {"cursor": cursor, "sort_order": "asc"}
                continue

            total_pages = response.metadata.total_pages
            if (
                "cursor" not in pagination
                and total_pages is not None
                and page < total_pages
            ):
                page += 1
                pagination = {"page": page, "sort_order": "asc"}
                continue

            if response.metadata.has_more is True:
                raise ValueError("Message listing has no usable pagination cursor")

            return messages
