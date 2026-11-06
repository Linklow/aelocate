"""AE source models, described by the spectrum of the out-of-plane source function.

Absolute amplitude is set separately (see :meth:`WaveformSimulator.waveforms`): a source's
``amplitude_db`` is the peak a sensor 10 cm away would record, which is how AE practitioners
talk about source strength (for example "a Hsu-Nielsen break gives ~95-100 dB near the sensor").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pulse_spectrum(g, duration: float, f: np.ndarray, n: int = 256) -> np.ndarray:
    """Fourier transform of a finite pulse g(t), t in [0, duration], by direct quadrature."""
    t = (np.arange(n) + 0.5) * duration / n
    dt = duration / n
    return (g(t)[None, :] * np.exp(-2j * np.pi * f[:, None] * t[None, :])).sum(axis=1) * dt


@dataclass(frozen=True)
class StepSource:
    """Step-like unloading with a raised-sine ramp: pencil-lead break or crack advance.

    The force drops over ``rise_time``. Shorter rise times give broader spectra: a Hsu-Nielsen
    pencil-lead break (0.5 mm 2H lead) is about 1-1.5 us, crack growth can be sub-microsecond.
    """

    name: str = "PLB"
    rise_time: float = 1.5e-6

    def spectrum(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        tr = self.rise_time

        def ramp_derivative(t):
            return (np.pi / (2 * tr)) * np.sin(np.pi * t / tr)  # integrates to 1 over [0, tr]

        g = _pulse_spectrum(ramp_derivative, tr, f)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0, g / (2j * np.pi * np.where(f > 0, f, 1.0)), 0.0)


@dataclass(frozen=True)
class BurstSource:
    """Gaussian tone burst: a pulser driving a transmitting transducer (a "location pulse").

    This is what an AE system emits during auto-sensor-test, and what a pulse generator injects
    when calibrating location on a structure.
    """

    name: str = "pulser"
    f0: float = 150e3
    cycles: float = 3.0

    def spectrum(self, f: np.ndarray) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        sigma = self.cycles / (2.0 * self.f0)
        duration = 8 * sigma

        def burst(t):
            return np.sin(2 * np.pi * self.f0 * (t - duration / 2)) * np.exp(-0.5 * ((t - duration / 2) / sigma) ** 2)

        return _pulse_spectrum(burst, duration, f, n=512)


PLB = StepSource("PLB", 1.5e-6)
CRACK = StepSource("crack", 0.6e-6)
PULSER = BurstSource("pulser", 150e3, 3.0)
