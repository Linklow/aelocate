"""Acquisition/processing worker that runs off the GUI thread.

The worker owns the simulated DAQ and the processing chain. It runs a paced loop
(acquire block -> detect hits -> build events -> locate with every method -> match to ground
truth) and emits one :class:`Frame` per block through a Qt signal. Qt queues the signal to the
GUI thread. Settings changed from the GUI are queued as commands and applied between blocks,
so the processing state is only ever touched by the worker thread.
"""

from __future__ import annotations

import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..daq import AcquisitionSimulator, Emitter, GroundTruthEvent
from ..events import EventBuilder
from ..hits import Hit, HitDetector
from ..location import LocatedEvent
from ..presets import Setup
from ..scenarios import default_emitters
from ..units import volts_to_dbae

ENVELOPE_BINS = 200


@dataclass
class Frame:
    t_start: float
    t_end: float
    env_t: np.ndarray
    envelope: np.ndarray
    """Min/max envelope, shape (n_channels, 2 * ENVELOPE_BINS), sensor volts."""
    hits: list[Hit]
    located: dict[str, list[LocatedEvent]]
    errors: dict[str, list[float]] = field(default_factory=dict)
    """Location error (m) of located events matched to a ground-truth source, per method."""
    truth: list[GroundTruthEvent] = field(default_factory=list)
    rms_db: np.ndarray | None = None
    load: float = 0.0


def envelope(data: np.ndarray, bins: int = ENVELOPE_BINS) -> np.ndarray:
    n_ch, n = data.shape
    usable = n - n % bins
    blocks = data[:, :usable].reshape(n_ch, bins, -1)
    out = np.empty((n_ch, 2 * bins), dtype=np.float32)
    out[:, 0::2] = blocks.min(axis=2)
    out[:, 1::2] = blocks.max(axis=2)
    return out


class AcquisitionWorker(QObject):
    frame = Signal(object)
    finished = Signal()

    def __init__(self, setup: Setup, locators: dict[str, object], block_seconds: float = 0.02,
                 seed: int | None = None, emitters: list[Emitter] | None = None) -> None:
        super().__init__()
        self.setup = setup
        self.block = int(round(setup.fs * block_seconds))
        self.speed = 1.0
        self._locators = dict(locators)
        self._commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._running = False
        self._paused = False
        self._truth: deque[GroundTruthEvent] = deque(maxlen=500)
        self.daq = AcquisitionSimulator(setup.simulator, emitters or default_emitters(setup),
                                        block_size=self.block, seed=seed)
        self.config = setup.detector
        self.detector = HitDetector(setup.n_channels, setup.fs, self.config)
        self.builder = EventBuilder(setup.event_window(), latency=self.detector.latency)

    # ---- commands (thread-safe: queued and executed by the worker between blocks) ----
    def submit(self, fn: Callable[[], None]) -> None:
        self._commands.put(fn)

    def set_threshold(self, db: float) -> None:
        def apply():
            self.config = replace(self.config, threshold_db=float(db))
            self.detector = HitDetector(self.setup.n_channels, self.setup.fs, self.config, self.daq.position)
        self.submit(apply)

    def set_rates(self, crack_hz: float, background_hz: float) -> None:
        def apply():
            em = [replace(e, rate_hz=crack_hz if e.name == "crack" else background_hz)
                  for e in self.daq.emitters]
            self.daq.set_emitters(em)
        self.submit(apply)

    def set_locator(self, name: str, locator) -> None:
        self.submit(lambda: self._locators.__setitem__(name, locator))

    def inject(self, source, position, amplitude_db: float) -> None:
        self.submit(lambda: self.daq.inject(source, position, amplitude_db, name="manual PLB"))

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def stop(self) -> None:
        self._running = False

    # ---- loop ----
    def _match(self, loc: LocatedEvent) -> float | None:
        t = loc.time
        best = None
        for ev in self._truth:
            if 0.0 <= t - ev.time <= 1e-3 and (best is None or ev.time > best.time):
                best = ev
        if best is None:
            return None
        return float(np.linalg.norm(np.subtract(loc.position, best.position[:len(loc.position)])))

    def step(self) -> Frame:
        """Acquire and process one block (also used directly by tests and screenshots)."""
        while True:
            try:
                self._commands.get_nowait()()
            except queue.Empty:
                break
        t0 = time.perf_counter()
        block = self.daq.next_block()
        self._truth.extend(block.events)
        hits = self.detector.process(block.data)
        self.builder.push(hits)
        events = self.builder.pop(block.end_time)
        located: dict[str, list[LocatedEvent]] = {}
        errors: dict[str, list[float]] = {}
        for name, loc in self._locators.items():
            located[name] = [r for r in (loc.locate(e) for e in events) if r is not None]
            errors[name] = [e for e in (self._match(r) for r in located[name]) if e is not None]
        data = block.data
        rms = volts_to_dbae(np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=1)))
        t_start = block.start / block.fs
        env_t = t_start + (np.arange(2 * ENVELOPE_BINS) // 2 + 0.5) * (data.shape[1] / ENVELOPE_BINS) / block.fs
        load = (time.perf_counter() - t0) / (data.shape[1] / block.fs)
        return Frame(t_start, block.end_time, env_t, envelope(data), hits, located, errors,
                     list(block.events), rms, load)

    def run(self) -> None:
        self._running = True
        deadline = time.perf_counter()
        while self._running:
            if self._paused:
                time.sleep(0.03)
                deadline = time.perf_counter()
                continue
            self.frame.emit(self.step())
            deadline += self.block / self.setup.fs / self.speed if np.isfinite(self.speed) else 0.0
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.perf_counter()
        self.finished.emit()
