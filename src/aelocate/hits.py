"""Streaming hit detection with standard AE timing parameters (PDT / HDT / HLT).

Definitions follow ASTM E1316 and the PAC/MISTRAS AEwin conventions:

* A **hit** starts at the first threshold crossing.
* **PDT** (peak definition time): if no higher peak arrives within PDT of the current peak,
  the peak is frozen. Rise time and counts-to-peak use the frozen peak.
* **HDT** (hit definition time): the hit ends once no threshold crossing occurs for HDT.
* **HLT** (hit lockout time): after a hit ends, crossings are ignored for HLT, which suppresses
  reflections and reverberation tails.

The detector consumes data block by block and keeps its state between blocks, so the hits do
not depend on the block size. Splitting a record into 1 block or 1000 gives the same hits, and
the unit tests check exactly that. Work is vectorised per *threshold excursion*, not per sample,
so it keeps up with multi-MHz multichannel streams in pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import ATTOJOULE, ENERGY_REF_OHM, dbae_to_volts, volts_to_dbae


@dataclass(frozen=True)
class HitDetectorConfig:
    """Hit-detection settings. Times are in seconds, buffer sizes in samples."""

    threshold_db: float = 45.0
    pdt: float = 200e-6
    hdt: float = 800e-6
    hlt: float = 1000e-6
    max_duration: float = 50e-3
    pretrigger: int = 512
    waveform_length: int = 2048

    def __post_init__(self) -> None:
        if self.waveform_length <= self.pretrigger:
            raise ValueError("waveform_length must be larger than pretrigger")


@dataclass(frozen=True, eq=False)
class Hit:
    """One AE hit with the classic feature set and its transient waveform."""

    channel: int
    index: int
    """Sample index of the first threshold crossing."""
    arrival_time: float
    """First threshold crossing, linearly interpolated between samples, s."""
    amplitude_db: float
    rise_time: float
    duration: float
    counts: int
    counts_to_peak: int
    energy_aj: float
    """Absolute energy, attojoules (integral of V^2 / 10 kOhm)."""
    waveform: np.ndarray = field(repr=False)
    """Sensor-referred volts, starting ``pretrigger`` samples before the first crossing."""
    waveform_start: int
    fs: float

    @property
    def pretrigger(self) -> int:
        return self.index - self.waveform_start

    @property
    def average_frequency(self) -> float:
        """Counts / duration, Hz."""
        return self.counts / self.duration if self.duration > 0 else 0.0

    def time_of_sample(self, i: float) -> float:
        """Absolute time of waveform sample ``i``."""
        return (self.waveform_start + i) / self.fs


@dataclass
class _OpenHit:
    start: int
    crossing: float
    last: int
    peak_val: float
    peak_idx: int
    counts: int
    counts_to_peak: int
    peak_locked: bool = False


class ChannelHitDetector:
    """Hit detector for a single channel. Feed it consecutive blocks with :meth:`process`."""

    def __init__(self, channel: int, fs: float, config: HitDetectorConfig | None = None,
                 start_index: int = 0) -> None:
        self.channel = channel
        self.fs = fs
        self.config = cfg = config or HitDetectorConfig()
        self.threshold = float(dbae_to_volts(cfg.threshold_db))
        self._pdt = int(round(cfg.pdt * fs))
        self._hdt = int(round(cfg.hdt * fs))
        self._hlt = int(round(cfg.hlt * fs))
        self._max_dur = int(round(cfg.max_duration * fs))
        self._buf = np.zeros(0)
        self._buf_start = start_index
        self._pos = start_index
        self._carry: int | None = None  # start of a threshold excursion still open at block end
        self._open: _OpenHit | None = None
        self._closed: list[_OpenHit] = []
        self._lockout_until = -1

    @property
    def position(self) -> int:
        """Global index of the next sample expected."""
        return self._pos

    def _get(self, a: int, b: int) -> np.ndarray:
        """Samples [a, b) by global index, zero-padded outside the buffered range."""
        out = np.zeros(b - a)
        lo, hi = max(a, self._buf_start), min(b, self._buf_start + self._buf.size)
        if hi > lo:
            out[lo - a:hi - a] = self._buf[lo - self._buf_start:hi - self._buf_start]
        return out

    def _crossing(self, g: int, polarity: int) -> float:
        """Sub-sample threshold-crossing position preceding global sample ``g``."""
        if g - 1 < self._buf_start:
            return float(g)
        y1 = polarity * self._buf[g - 1 - self._buf_start]
        y2 = polarity * self._buf[g - self._buf_start]
        frac = (self.threshold - y1) / (y2 - y1) if y2 > y1 else 1.0
        return g - 1 + float(np.clip(frac, 0.0, 1.0))

    def _close(self) -> None:
        h = self._open
        assert h is not None
        self._closed.append(h)
        self._lockout_until = h.last + self._hdt + self._hlt
        self._open = None

    def _finalize(self, h: _OpenHit) -> Hit:
        cfg = self.config
        seg = self._get(h.start, h.last + 1)
        energy = float(np.sum(seg * seg)) / self.fs / ENERGY_REF_OHM / ATTOJOULE
        w0 = h.start - cfg.pretrigger
        return Hit(
            channel=self.channel,
            index=h.start,
            arrival_time=h.crossing / self.fs,
            amplitude_db=float(volts_to_dbae(h.peak_val)),
            rise_time=(h.peak_idx - h.start) / self.fs,
            duration=(h.last - h.start) / self.fs,
            counts=h.counts,
            counts_to_peak=h.counts_to_peak,
            energy_aj=energy,
            waveform=self._get(w0, w0 + cfg.waveform_length).astype(np.float32),
            waveform_start=w0,
            fs=self.fs,
        )

    def _runs(self, b0: int, n: int):
        """Complete threshold excursions in the buffer that end inside the current block.

        Yields (global_start, global_end_inclusive, polarity, peak_value, global_peak_index).
        Excursions still open at the block end are carried over to the next call.
        """
        scan_from = self._carry if self._carry is not None else b0
        x = self._buf[scan_from - self._buf_start:]
        thr = self.threshold
        state = np.zeros(x.size, dtype=np.int8)
        state[x >= thr] = 1
        state[x <= -thr] = -1
        if state.size == 0:
            return
        prev = np.concatenate(([0], state[:-1]))
        nxt = np.concatenate((state[1:], [0]))
        starts = np.flatnonzero((state != 0) & (state != prev))
        ends = np.flatnonzero((state != 0) & (state != nxt))
        self._carry = None
        if state[-1] != 0:  # last excursion continues into the next block
            self._carry = scan_from + int(starts[-1])
            starts, ends = starts[:-1], ends[:-1]
        absx = np.abs(x)
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            i = s + int(np.argmax(absx[s:e + 1]))
            yield scan_from + s, scan_from + e, int(state[s]), float(absx[i]), scan_from + i

    def process(self, block: np.ndarray) -> list[Hit]:
        """Process the next block of samples (sensor-referred volts) and return finished hits."""
        x = np.asarray(block, dtype=float).ravel()
        b0 = self._pos
        self._buf = np.concatenate((self._buf, x))
        self._pos += x.size

        for gs, ge, pol, pv, pidx in self._runs(b0, x.size):
            h = self._open
            if h is not None and (gs - h.last > self._hdt or gs - h.start > self._max_dur):
                self._close()
                h = None
            if h is None:
                if gs <= self._lockout_until:
                    continue
                c = int(pol > 0)
                self._open = _OpenHit(gs, self._crossing(gs, pol), ge, pv, pidx, c, c)
                continue
            h.last = ge
            if pol > 0:
                h.counts += 1
            if not h.peak_locked:
                if pidx - h.peak_idx > self._pdt:
                    h.peak_locked = True
                elif pv > h.peak_val:
                    h.peak_val, h.peak_idx, h.counts_to_peak = pv, pidx, h.counts

        h = self._open
        if h is not None and self._carry is None:
            now = self._pos - 1
            if now - h.last > self._hdt or now - h.start > self._max_dur:
                self._close()

        return self._emit_ready()

    def _emit_ready(self, force: bool = False) -> list[Hit]:
        cfg = self.config
        out, waiting = [], []
        for h in self._closed:
            if force or self._pos >= h.start - cfg.pretrigger + cfg.waveform_length:
                out.append(self._finalize(h))
            else:
                waiting.append(h)
        self._closed = waiting
        self._trim()
        return out

    def _trim(self) -> None:
        keep = [self._pos]
        if self._open is not None:
            keep.append(self._open.start)
        if self._carry is not None:
            keep.append(self._carry)
        keep.extend(h.start for h in self._closed)
        keep_from = max(min(keep) - self.config.pretrigger - 2, self._buf_start)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start:]
            self._buf_start = keep_from

    def flush(self) -> list[Hit]:
        """End of stream: close any open hit and return everything pending (zero-padded)."""
        if self._carry is not None:
            # Treat the open excursion as ending at the last sample.
            end = self._pos - 1
            seg = np.abs(self._buf[self._carry - self._buf_start:])
            i = int(np.argmax(seg))
            pol = 1 if self._buf[self._carry - self._buf_start] > 0 else -1
            if self._open is None and self._carry > self._lockout_until:
                c = int(pol > 0)
                self._open = _OpenHit(self._carry, self._crossing(self._carry, pol), end,
                                      float(seg[i]), self._carry + i, c, c)
            elif self._open is not None:
                self._open.last = end
            self._carry = None
        if self._open is not None:
            self._close()
        return self._emit_ready(force=True)


class HitDetector:
    """Multichannel hit detector: one :class:`ChannelHitDetector` per channel."""

    def __init__(self, n_channels: int, fs: float, config: HitDetectorConfig | None = None,
                 start_index: int = 0) -> None:
        self.config = config or HitDetectorConfig()
        self.fs = fs
        self.channels = [ChannelHitDetector(i, fs, self.config, start_index) for i in range(n_channels)]

    def process(self, block: np.ndarray) -> list[Hit]:
        """Process an ``(n_channels, n_samples)`` block; returns hits sorted by arrival time."""
        block = np.atleast_2d(block)
        if block.shape[0] != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} channels, got {block.shape[0]}")
        hits = [h for det, x in zip(self.channels, block, strict=True) for h in det.process(x)]
        return sorted(hits, key=lambda h: h.arrival_time)

    def flush(self) -> list[Hit]:
        hits = [h for det in self.channels for h in det.flush()]
        return sorted(hits, key=lambda h: h.arrival_time)

    @property
    def latency(self) -> float:
        """Upper bound on the delay between a hit's arrival and its emission, s."""
        c = self.config
        return c.max_duration + c.hdt + c.waveform_length / self.fs + 1e-3
