"""Simulated multichannel AE acquisition: continuous streaming like a real DAQ board.

:class:`AcquisitionSimulator` produces consecutive blocks of digitised, sensor-referred samples
on every channel: band-limited background noise plus the waveforms of AE events from a set of
:class:`Emitter` populations (for example "a growing crack at x = 0.62 m" plus "random
friction noise near a support"). Each block also carries the ground truth of the events it
contains, so every processing stage can be checked against the true positions.

:func:`simulate_event` synthesises one isolated event (for datasets and benchmarks).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal

from .instrument import FrontEnd
from .simulator import WaveformSimulator


class NoiseGenerator:
    """Gaussian background noise coloured by the front-end filter, continuous across calls."""

    def __init__(self, frontend: FrontEnd, n_channels: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.n_channels = n_channels
        self.sos = frontend.digital_filter_sos()
        impulse = np.zeros(32768)
        impulse[0] = 1.0
        noise_gain = float(np.sqrt(np.sum(signal.sosfilt(self.sos, impulse) ** 2)))
        self.scale = frontend.noise_uv_rms * 1e-6 / noise_gain
        self.zi = np.zeros((self.sos.shape[0], n_channels, 2))
        self.generate(4096)  # settle the filter state

    def generate(self, n: int) -> np.ndarray:
        if self.scale == 0:
            return np.zeros((self.n_channels, n))
        white = self.rng.standard_normal((self.n_channels, n)) * self.scale
        out, self.zi = signal.sosfilt(self.sos, white, axis=-1, zi=self.zi)
        return out


@dataclass(frozen=True)
class Emitter:
    """A population of AE sources emitting as a Poisson process.

    Attributes:
        name: Label used in the ground truth ("crack", "friction", ...).
        source: Source model.
        rate_hz: Mean event rate.
        center: Mean position (x,) or (x, y), m.
        spread: Gaussian sigma (``distribution="normal"``) or half-width (``"uniform"``), m.
        amplitude_db: Uniform range of source amplitude (dB_AE at the reference distance).
    """

    name: str
    source: object
    rate_hz: float
    center: tuple[float, ...]
    spread: float = 0.0
    distribution: str = "normal"
    amplitude_db: tuple[float, float] = (50.0, 70.0)

    def sample_position(self, rng: np.random.Generator) -> np.ndarray:
        c = np.asarray(self.center, dtype=float)
        if self.distribution == "uniform":
            return c + rng.uniform(-self.spread, self.spread, c.size)
        return c + rng.normal(0.0, self.spread, c.size)

    def sample_amplitude(self, rng: np.random.Generator) -> float:
        lo, hi = self.amplitude_db
        return float(rng.uniform(lo, hi))


@dataclass(frozen=True)
class GroundTruthEvent:
    time: float
    position: tuple[float, ...]
    amplitude_db: float
    emitter: str
    source: str


@dataclass(frozen=True, eq=False)
class AcquisitionBlock:
    start: int
    """Global index of the first sample."""
    fs: float
    data: np.ndarray
    """Sensor-referred volts, shape (n_channels, n_samples), float32."""
    events: tuple[GroundTruthEvent, ...]

    @property
    def end_time(self) -> float:
        return (self.start + self.data.shape[1]) / self.fs


class AcquisitionSimulator:
    """Continuous multichannel stream generator.

    Args:
        simulator: Waveform simulator (specimen, sensor, front-end).
        emitters: Source populations.
        block_size: Samples per channel per block.
        seed: Random seed.
    """

    def __init__(self, simulator: WaveformSimulator, emitters: Sequence[Emitter] = (),
                 block_size: int = 40_000, seed: int | None = None) -> None:
        self.sim = simulator
        self.fs = simulator.fs
        self.n_channels = simulator.n_channels
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.noise = NoiseGenerator(simulator.frontend, self.n_channels, self.rng)
        self.position = 0
        self._active: list[tuple[int, np.ndarray]] = []
        self._injected: list[GroundTruthEvent] = []
        self._emitters: list[Emitter] = []
        self._next_time: list[float] = []
        self.set_emitters(emitters)

    @property
    def time(self) -> float:
        return self.position / self.fs

    @property
    def emitters(self) -> tuple[Emitter, ...]:
        return tuple(self._emitters)

    def set_emitters(self, emitters: Sequence[Emitter]) -> None:
        """Replace the source populations (takes effect from the current stream time)."""
        self._emitters = list(emitters)
        self._next_time = [self._draw_next(e, self.time) for e in self._emitters]

    def _draw_next(self, e: Emitter, t: float) -> float:
        return t + self.rng.exponential(1.0 / e.rate_hz) if e.rate_hz > 0 else np.inf

    def inject(self, source, position, amplitude_db: float, t0: float | None = None,
               name: str = "manual") -> GroundTruthEvent:
        """Inject a single event (e.g. a pencil-lead break from the GUI) at time ``t0``."""
        t0 = self.time + 1e-4 if t0 is None else t0
        pos = self.sim.specimen.clamp(position)
        start, w = self.sim.waveforms(source, pos, amplitude_db, t0)
        self._active.append((start, w))
        ev = GroundTruthEvent(t0, tuple(float(p) for p in pos), amplitude_db, name, source.name)
        self._injected.append(ev)
        return ev

    def next_block(self, n: int | None = None) -> AcquisitionBlock:
        n = n or self.block_size
        a, b = self.position, self.position + n
        t_end = b / self.fs
        truth, self._injected = self._injected, []
        for i, e in enumerate(self._emitters):
            while self._next_time[i] < t_end:
                t0 = self._next_time[i]
                pos = self.sim.specimen.clamp(e.sample_position(self.rng))
                amp = e.sample_amplitude(self.rng)
                self._active.append(self.sim.waveforms(e.source, pos, amp, t0))
                truth.append(GroundTruthEvent(t0, tuple(float(p) for p in pos), amp, e.name, e.source.name))
                self._next_time[i] = self._draw_next(e, t0)

        data = self.noise.generate(n)
        still_active = []
        for start, w in self._active:
            end = start + w.shape[1]
            lo, hi = max(a, start), min(b, end)
            if hi > lo:
                data[:, lo - a:hi - a] += w[:, lo - start:hi - start]
            if end > b:
                still_active.append((start, w))
        self._active = still_active
        data = self.sim.frontend.digitize(data).astype(np.float32)
        self.position = b
        truth.sort(key=lambda ev: ev.time)
        return AcquisitionBlock(a, self.fs, data, tuple(truth))


def simulate_event(sim: WaveformSimulator, source, position, amplitude_db: float,
                   noise: NoiseGenerator | None = None, t0: float = 0.0,
                   pad_before: int = 1024) -> tuple[int, np.ndarray]:
    """One isolated, digitised event (noise included if ``noise`` is given).

    ``pad_before`` samples of background are prepended so the pre-trigger of every hit holds
    real noise, as it would in a continuous recording.

    Returns:
        ``(start_index, data)`` with ``data`` of shape (n_channels, pad_before + sim.n_fft);
        ``data[:, i]`` is sample ``start_index + i`` of a stream whose t=0 is the source time
        reference.
    """
    start, w = sim.waveforms(source, position, amplitude_db, t0)
    data = np.zeros((w.shape[0], pad_before + w.shape[1]))
    data[:, pad_before:] = w
    if noise is not None:
        data += noise.generate(data.shape[1])
    return start - pad_before, sim.frontend.digitize(data)
