"""Frequency-domain synthesis of multichannel AE waveforms.

For a source at position ``x`` and a sensor at distance ``r`` along each propagation path:

    U(f) = S(f) * H_sensor(f) * H_filter(f)
           * sum_paths g * (r_ref / r_eff)^n * exp(-alpha(f) r)
           * sum_modes w_m * exp(-i k_m(f) r)

``k_m(f)`` comes from the Rayleigh-Lamb solution, so dispersion, mode separation and the
amplitude-dependent threshold crossing come out of the physics instead of a fixed delay.
"""

from __future__ import annotations

import numpy as np

from .dispersion import MODES, DispersionCurve, lamb_dispersion
from .instrument import FrontEnd, Sensor
from .specimen import Specimen
from .units import db_per_m_to_np_per_m, dbae_to_volts


class WaveformSimulator:
    """Synthesises sensor-referred voltage waveforms for an AE source on a specimen.

    Args:
        specimen: Test object and sensor layout.
        sensor: Sensor model (same on every channel).
        frontend: Preamp/filter/ADC model.
        n_fft: Length of each synthesised event window, samples.
        ref_distance: Distance at which ``amplitude_db`` is defined, m.
        min_distance: Near-field regularisation for geometric spreading, m.
        lead_samples: Samples kept before the earliest possible arrival.
    """

    def __init__(
        self,
        specimen: Specimen,
        sensor: Sensor | None = None,
        frontend: FrontEnd | None = None,
        n_fft: int = 8192,
        ref_distance: float = 0.1,
        min_distance: float = 0.02,
        lead_samples: int = 128,
    ) -> None:
        self.specimen = specimen
        self.sensor = sensor or Sensor()
        self.frontend = frontend or FrontEnd()
        self.fs = self.frontend.fs
        self.n_fft = n_fft
        self.ref_distance = ref_distance
        self.min_distance = min_distance
        self.lead_samples = lead_samples

        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / self.fs)
        self.curves: dict[str, DispersionCurve] = lamb_dispersion(
            specimen.material, specimen.thickness, f_max=max(self.fs / 2, 1.2e6))
        self._k = {m: self.curves[m].wavenumber(self.freqs) for m in MODES}
        self._weights = {"A0": 1.0, "S0": specimen.s0_ratio}
        self._chain = self.sensor.response(self.freqs) * self.frontend.filter_response(self.freqs)
        self._alpha = db_per_m_to_np_per_m(specimen.attenuation_db_per_m) * self.freqs / specimen.attenuation_f_ref
        self._c_fastest = float(max(self.curves["S0"].group_velocity.max(), self.curves["A0"].group_velocity.max()))
        taper = np.ones(n_fft)
        n_tap = n_fft // 10
        taper[-n_tap:] = np.cos(np.linspace(0, np.pi / 2, n_tap)) ** 2
        self._taper = taper
        self._source_cache: dict[object, np.ndarray] = {}
        self._scale_cache: dict[object, float] = {}

    @property
    def n_channels(self) -> int:
        return self.specimen.n_channels

    def _source_spectrum(self, source) -> np.ndarray:
        if source not in self._source_cache:
            self._source_cache[source] = source.spectrum(self.freqs) * self._chain
        return self._source_cache[source]

    def _impulse(self, source, paths, delay: float) -> np.ndarray:
        prop = np.zeros_like(self.freqs, dtype=complex)
        for r, g in paths:
            r_eff = np.hypot(r, self.min_distance)
            amp = g * (self.ref_distance / r_eff) ** self.specimen.spreading_exponent * np.exp(-self._alpha * r)
            for mode in MODES:
                w = self._weights[mode]
                if w:
                    prop += w * amp * np.exp(-1j * self._k[mode] * r)
        spec = self._source_spectrum(source) * prop * np.exp(-2j * np.pi * self.freqs * delay)
        return np.fft.irfft(spec, self.n_fft) * self._taper

    def _unit_scale(self, source) -> float:
        """Scale that makes the peak at ``ref_distance`` equal to 1 V (before applying amplitude)."""
        if source not in self._scale_cache:
            w = self._impulse(source, [(self.ref_distance, 1.0)], self.lead_samples / self.fs)
            self._scale_cache[source] = 1.0 / float(np.max(np.abs(w)))
        return self._scale_cache[source]

    def waveforms(self, source, position, amplitude_db: float, t0: float) -> tuple[int, np.ndarray]:
        """Waveforms on all channels for one AE event.

        Args:
            source: Source model (:class:`~aelocate.sources.StepSource` or ``BurstSource``).
            position: Source coordinate(s), m.
            amplitude_db: Peak amplitude a sensor at ``ref_distance`` would record, dB_AE.
            t0: Source time, s.

        Returns:
            ``(start_index, w)``: ``w[ch, n]`` is the sensor voltage at sample ``start_index + n``.
        """
        paths = [self.specimen.paths(position, ch) for ch in range(self.n_channels)]
        r_min = min(r for ch_paths in paths for r, _ in ch_paths)
        start = int(np.floor((t0 + r_min / self._c_fastest) * self.fs)) - self.lead_samples
        delay = t0 - start / self.fs
        scale = float(dbae_to_volts(amplitude_db)) * self._unit_scale(source)
        w = np.stack([self._impulse(source, p, delay) for p in paths]) * scale
        return start, w

    def arrival_times(self, position, t0: float = 0.0, mode: str = "A0", f: float | None = None) -> np.ndarray:
        """Direct-path group arrival time on every channel for a mode at frequency ``f``."""
        f = self.sensor.f_res if f is None else f
        cg = float(self.curves[mode].group_at(f))
        r = np.array([self.specimen.paths(position, ch)[0][0] for ch in range(self.n_channels)])
        return t0 + r / cg
