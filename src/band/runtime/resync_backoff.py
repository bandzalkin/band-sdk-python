"""How long an idle room waits before re-polling /next.

The idle poll is only a safety net for a WebSocket push that never arrived,
and an agent runs one per room, so its REST cost grows with the number of
rooms. A room whose polls keep finding nothing to run doubles its interval each
time, up to a cap; an event queued for the room, or a backlog message the room
claims, brings it back to the base interval. Each wait is drawn from the upper
half of the interval, so rooms that started together (every room an agent joins
at startup) drift apart instead of polling in the same instant.
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable


class IdleResyncBackoff:
    """One room's idle /next interval."""

    def __init__(
        self,
        base_seconds: float,
        max_seconds: float,
        *,
        random: Callable[[], float] = _random.random,
    ) -> None:
        self._base = base_seconds
        # A cap below the base would shorten the interval it is meant to bound.
        self._max = max(max_seconds, base_seconds)
        self._random = random
        self._level = base_seconds
        # Lets a caller tell whether traffic was seen while a poll ran.
        self.resets = 0

    @property
    def level(self) -> float:
        """The current interval, before jitter."""
        return self._level

    def next_wait(self) -> float:
        """Seconds to wait for a push before the next idle poll."""
        return self._level * (0.5 + 0.5 * self._random())

    def found_nothing(self) -> None:
        """An idle poll found nothing to run: wait longer next time."""
        self._level = min(self._level * 2, self._max)

    def reset(self) -> None:
        """The room has traffic: poll at the base interval again."""
        self._level = self._base
        self.resets += 1
