"""Measurement chain: AE sensor, preamplifier, analog band-pass filter and ADC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from .units import db_to_ratio


@dataclass(frozen=True)
class Sensor:
    """Piezoelectric AE sensor modelled by its frequency response.

    ``resonant`` is a single-resonance band-pass (e.g. a 150 kHz sensor commonly used on steel);
    ``wideband`` is a flat Butterworth band (e.g. 100 kHz - 1 MHz).
    """

    name: str = "150 kHz resonant"
    kind: Literal["resonant", "wideband"] = "resonant"
    f_res: float = 150e3
    q_factor: float = 5.0
    band: tuple[float, float] = (100e3, 1.0e6)

    def response(self, f) -> np.ndarray:
        """Complex frequency response (peak gain 1)."""
        w = 2 * np.pi * np.asarray(f, dtype=float)
        if self.kind == "resonant":
            w0 = 2 * np.pi * self.f_res
            s = 1j * w
            return (s * w0 / self.q_factor) / (s**2 + s * w0 / self.q_factor + w0**2)
        z, p, k = signal.butter(2, [2 * np.pi * self.band[0], 2 * np.pi * self.band[1]], "bandpass",
                                analog=True, output="zpk")
        return signal.freqs_zpk(z, p, k, worN=w)[1]


@dataclass(frozen=True)
class FrontEnd:
    """Preamplifier + analog filter + digitiser of one acquisition channel.

    Attributes:
        fs: Sampling rate, Hz.
        preamp_gain_db: Preamplifier gain (40 dB is the de-facto standard).
        band: Analog band-pass filter edges, Hz.
        filter_order: Butterworth order per band edge.
        full_scale: ADC input range (+/- volts, after the preamplifier).
        bits: ADC resolution.
        noise_uv_rms: Sensor-referred background noise, uV RMS, inside the filter band.
    """

    fs: float = 2e6
    preamp_gain_db: float = 40.0
    band: tuple[float, float] = (100e3, 400e3)
    filter_order: int = 4
    full_scale: float = 10.0
    bits: int = 16
    noise_uv_rms: float = 15.0

    @property
    def gain(self) -> float:
        return float(db_to_ratio(self.preamp_gain_db))

    @property
    def clip_level(self) -> float:
        """Largest representable sensor-referred voltage (100 dB_AE for 40 dB gain and 10 V)."""
        return self.full_scale / self.gain

    @property
    def lsb(self) -> float:
        """Sensor-referred quantisation step, volts."""
        return 2 * self.full_scale / 2**self.bits / self.gain

    def filter_response(self, f) -> np.ndarray:
        """Complex response of the analog band-pass filter."""
        z, p, k = signal.butter(self.filter_order, [2 * np.pi * self.band[0], 2 * np.pi * self.band[1]],
                                "bandpass", analog=True, output="zpk")
        return signal.freqs_zpk(z, p, k, worN=2 * np.pi * np.asarray(f, dtype=float))[1]

    def digital_filter_sos(self) -> np.ndarray:
        """Digital equivalent of the analog filter, used to colour the background noise."""
        return signal.butter(self.filter_order, list(self.band), "bandpass", fs=self.fs, output="sos")

    def digitize(self, x: np.ndarray) -> np.ndarray:
        """Clip to the ADC range and quantise (sensor-referred volts in, sensor-referred volts out)."""
        lsb = self.lsb
        return np.round(np.clip(x, -self.clip_level, self.clip_level) / lsb) * lsb
