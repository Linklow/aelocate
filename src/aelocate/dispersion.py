"""Lamb-wave dispersion of the fundamental modes (A0, S0) in an isotropic plate.

The Rayleigh-Lamb characteristic equations are solved numerically for the phase velocity at
each frequency. AE in plates, strips, pipes and vessels propagates mostly as these two modes:

* **S0** (extensional): fast (~5.4 km/s in steel), weakly dispersive, small out-of-plane
  amplitude, so it often stays *below* the detection threshold.
* **A0** (flexural): slower (~3 km/s at 150 kHz in 5 mm steel), strongly dispersive, carries
  most of the out-of-plane energy a resonant AE sensor sees.

Which mode crosses the threshold first depends on signal amplitude and distance. That
mode switching is one of the main sources of error in the standard linear-location algorithm,
and the reason this package models both modes instead of using a single wave speed.

Numerics: with half-thickness ``h``, wavenumber ``k`` and ``p^2 = w^2/c_l^2 - k^2``,
``q^2 = w^2/c_t^2 - k^2``, the symmetric and antisymmetric equations are rewritten as

    F_S = (q^2/k^2 - 1)^2 cos(ph) sinc(qh) + 4 (p^2/k^2) sinc(ph) cos(qh)
    F_A = (q^2/k^2 - 1)^2 sinc(ph) cos(qh) + 4 (q^2/k^2) cos(ph) sinc(qh)

with ``sinc(x) = sin(x)/x``. Both are real for real *and* imaginary ``p, q``. For evanescent
components the terms are divided by ``cosh(|x|h)``; that positive factor keeps the sign, which
is all a bracketing root finder needs, and avoids overflow at large ``kh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .materials import Material

MODES = ("A0", "S0")


def _cos_sinc(x2: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """cos(xh) and sin(xh)/(xh) for x = sqrt(x2), normalised by cosh(|x|h) if x2 < 0."""
    x2 = np.asarray(x2, dtype=float)
    xh = np.sqrt(np.abs(x2)) * h
    propagating = x2 >= 0
    safe = np.where(xh > 0, xh, 1.0)
    cos_term = np.where(propagating, np.cos(xh), 1.0)
    sinc_term = np.where(propagating, np.sinc(xh / np.pi), np.where(xh > 0, np.tanh(xh) / safe, 1.0))
    return cos_term, sinc_term


def _terms(c, omega: float, mat: Material, h: float):
    c = np.asarray(c, dtype=float)
    k2 = (omega / c) ** 2
    p2_k2 = c**2 / mat.c_l**2 - 1.0
    q2_k2 = c**2 / mat.c_t**2 - 1.0
    cp, sp = _cos_sinc(k2 * p2_k2, h)
    cq, sq = _cos_sinc(k2 * q2_k2, h)
    a = (q2_k2 - 1.0) ** 2
    return a, p2_k2, q2_k2, cp, sp, cq, sq


def char_symmetric(c, omega: float, mat: Material, h: float) -> np.ndarray:
    """Normalised symmetric Rayleigh-Lamb function; zero at symmetric-mode phase velocities."""
    a, p2_k2, _, cp, sp, cq, sq = _terms(c, omega, mat, h)
    return a * cp * sq + 4.0 * p2_k2 * sp * cq


def char_antisymmetric(c, omega: float, mat: Material, h: float) -> np.ndarray:
    """Normalised antisymmetric Rayleigh-Lamb function; zero at antisymmetric-mode phase velocities."""
    a, _, q2_k2, cp, sp, cq, sq = _terms(c, omega, mat, h)
    return a * sp * cq + 4.0 * q2_k2 * cp * sq


@dataclass(frozen=True, eq=False)
class DispersionCurve:
    """Phase and group velocity of one Lamb mode tabulated over frequency."""

    mode: str
    freqs: np.ndarray
    phase_velocity: np.ndarray
    group_velocity: np.ndarray

    def _interp(self, f, values: np.ndarray, low_exponent: float) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        logf = np.log(np.maximum(f, 1e-12))
        out = np.interp(logf, np.log(self.freqs), values)
        below = f < self.freqs[0]
        if np.any(below):
            # Low-frequency asymptotes: A0 ~ sqrt(f) (Kirchhoff plate), S0 ~ constant.
            scale = (np.maximum(f, 0.0) / self.freqs[0]) ** low_exponent
            out = np.where(below, values[0] * scale, out)
        return out

    def phase_at(self, f) -> np.ndarray:
        """Phase velocity at frequency ``f`` (Hz), m/s."""
        return self._interp(f, self.phase_velocity, 0.5 if self.mode.startswith("A") else 0.0)

    def group_at(self, f) -> np.ndarray:
        """Group velocity at frequency ``f`` (Hz), m/s."""
        return self._interp(f, self.group_velocity, 0.5 if self.mode.startswith("A") else 0.0)

    def wavenumber(self, f) -> np.ndarray:
        """Wavenumber ``k = 2*pi*f / c_phase`` (rad/m); zero at DC."""
        f = np.asarray(f, dtype=float)
        c = self.phase_at(f)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(f > 0, 2 * np.pi * f / np.where(c > 0, c, 1.0), 0.0)


def _lowest_root(fn, omega: float, mat: Material, h: float, c_grid: np.ndarray) -> float:
    values = fn(c_grid, omega, mat, h)
    changes = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if changes.size == 0:
        raise RuntimeError(f"no Lamb-mode root found at f={omega / 2 / np.pi:.0f} Hz")
    j = changes[0]
    return float(brentq(lambda c: float(fn(c, omega, mat, h)), c_grid[j], c_grid[j + 1], xtol=1e-7))


@lru_cache(maxsize=32)
def lamb_dispersion(
    material: Material,
    thickness: float,
    f_max: float = 1.2e6,
    f_min: float = 5e3,
    n_freqs: int = 160,
) -> dict[str, DispersionCurve]:
    """Solve the Rayleigh-Lamb equations for the A0 and S0 modes.

    Args:
        material: Plate material.
        thickness: Plate thickness, m.
        f_max: Highest tabulated frequency, Hz.
        f_min: Lowest tabulated frequency, Hz. Below it the low-frequency asymptotes are used.
        n_freqs: Number of log-spaced frequencies.

    Returns:
        ``{"A0": DispersionCurve, "S0": DispersionCurve}``.
    """
    h = thickness / 2.0
    freqs = np.geomspace(f_min, f_max, n_freqs)
    # A0 phase velocity lies in (0, c_R); S0 lies in (c_R, c_plate].
    searches = {
        "A0": (char_antisymmetric, np.geomspace(0.02 * material.c_t, 0.9999 * material.c_t, 1500)),
        "S0": (char_symmetric, np.geomspace(0.98 * material.c_rayleigh, 0.9999 * material.c_l, 1500)),
    }
    curves: dict[str, DispersionCurve] = {}
    omega = 2 * np.pi * freqs
    for mode, (fn, grid) in searches.items():
        cp = np.array([_lowest_root(fn, w, material, h, grid) for w in omega])
        k = omega / cp
        cg = np.gradient(omega, k)
        curves[mode] = DispersionCurve(mode, freqs, cp, cg)
    return curves


def kirchhoff_a0_phase_velocity(material: Material, thickness: float, f) -> np.ndarray:
    """Classical (Kirchhoff) thin-plate flexural phase velocity, valid for f*d << c_t."""
    return np.sqrt(2 * np.pi * np.asarray(f, dtype=float) * material.c_plate * thickness / np.sqrt(12.0))
