"""Grouping hits from several channels into AE events."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .hits import Hit


@dataclass(frozen=True, eq=False)
class Event:
    """Hits that belong to one AE source event: the first hit on each channel, in arrival order."""

    hits: tuple[Hit, ...]

    @property
    def first(self) -> Hit:
        return self.hits[0]

    @property
    def time(self) -> float:
        return self.hits[0].arrival_time

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(h.channel for h in self.hits)

    def hit_on(self, channel: int) -> Hit | None:
        for h in self.hits:
            if h.channel == channel:
                return h
        return None


class EventBuilder:
    """Groups hits into events with an event-definition window.

    The first hit opens an event. The first hit of every other channel that arrives within
    ``window`` joins it, and later hits on a channel already in the event are discarded
    (reflections, late modes). For streaming use, hits are held for ``latency`` seconds
    because a hit on one channel can finish (and be reported) after a hit that arrived later
    on another channel.

    Args:
        window: Event definition time, s. Usually 1.1-1.3x the longest sensor-to-sensor
            travel time at the slowest relevant wave speed.
        latency: How long to wait for late-reported hits, s.
    """

    def __init__(self, window: float, latency: float = 0.0) -> None:
        self.window = window
        self.latency = latency
        self._pending: list[Hit] = []

    def push(self, hits: Iterable[Hit]) -> None:
        self._pending.extend(hits)
        self._pending.sort(key=lambda h: h.arrival_time)

    def pop(self, now: float | None = None) -> list[Event]:
        """Return the events that are complete at stream time ``now`` (all of them if ``None``)."""
        pending = self._pending
        events: list[Event] = []
        i = 0
        while i < len(pending):
            t1 = pending[i].arrival_time
            if now is not None and t1 + self.window + self.latency > now:
                break
            group: dict[int, Hit] = {pending[i].channel: pending[i]}
            j = i + 1
            while j < len(pending) and pending[j].arrival_time <= t1 + self.window:
                group.setdefault(pending[j].channel, pending[j])
                j += 1
            events.append(Event(tuple(sorted(group.values(), key=lambda h: h.arrival_time))))
            i = j
        self._pending = pending[i:]
        return events
