"""Streaming processing chain: block -> hits -> events -> locations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .daq import AcquisitionBlock
from .events import Event, EventBuilder
from .hits import Hit, HitDetector, HitDetectorConfig
from .location import LocatedEvent


@dataclass
class ChainOutput:
    hits: list[Hit] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    located: list[LocatedEvent] = field(default_factory=list)

    @property
    def unlocated(self) -> int:
        return len(self.events) - len(self.located)


class ProcessingChain:
    """Hit detection, event building and location for a continuous stream.

    Args:
        n_channels: Channel count.
        fs: Sampling rate, Hz.
        locator: Any object with ``locate(event) -> LocatedEvent | None``.
        event_window: Event definition time, s.
        detector: Hit-detection settings.
    """

    def __init__(self, n_channels: int, fs: float, locator, event_window: float,
                 detector: HitDetectorConfig | None = None) -> None:
        self.detector = HitDetector(n_channels, fs, detector)
        self.builder = EventBuilder(event_window, latency=self.detector.latency)
        self.locator = locator
        self.fs = fs

    @classmethod
    def from_setup(cls, setup, locator) -> ProcessingChain:
        return cls(setup.n_channels, setup.fs, locator, setup.event_window(), setup.detector)

    def _locate(self, events: list[Event], out: ChainOutput) -> ChainOutput:
        out.events = events
        if self.locator is not None:
            out.located = [loc for loc in (self.locator.locate(e) for e in events) if loc is not None]
        return out

    def process(self, block: AcquisitionBlock) -> ChainOutput:
        hits = self.detector.process(block.data)
        self.builder.push(hits)
        return self._locate(self.builder.pop(block.end_time), ChainOutput(hits))

    def flush(self) -> ChainOutput:
        hits = self.detector.flush()
        self.builder.push(hits)
        return self._locate(self.builder.pop(None), ChainOutput(hits))
