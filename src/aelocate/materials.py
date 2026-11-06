"""Isotropic elastic materials used by the wave-propagation model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Material:
    """Isotropic material described by density and bulk wave velocities.

    Attributes:
        name: Human-readable name.
        density: Density, kg/m^3.
        c_l: Longitudinal (P) bulk wave velocity, m/s.
        c_t: Transverse (S) bulk wave velocity, m/s.
    """

    name: str
    density: float
    c_l: float
    c_t: float

    def __post_init__(self) -> None:
        if not 0 < self.c_t < self.c_l / np.sqrt(2) * 1.0001:
            # c_t < c_l / sqrt(2) is equivalent to a non-negative Poisson ratio.
            raise ValueError(f"unphysical velocities for {self.name}: c_l={self.c_l}, c_t={self.c_t}")

    @property
    def poisson(self) -> float:
        """Poisson ratio derived from the bulk velocities."""
        cl2, ct2 = self.c_l**2, self.c_t**2
        return (cl2 - 2 * ct2) / (2 * (cl2 - ct2))

    @property
    def c_plate(self) -> float:
        """Low-frequency limit of the S0 Lamb mode (plate velocity), m/s."""
        return 2 * self.c_t * np.sqrt(1 - (self.c_t / self.c_l) ** 2)

    @cached_property
    def c_rayleigh(self) -> float:
        """Rayleigh surface-wave velocity (exact root of the Rayleigh equation), m/s."""
        kappa = (self.c_t / self.c_l) ** 2

        def rayleigh(xi: float) -> float:
            return (2 - xi) ** 2 - 4 * np.sqrt(1 - xi * kappa) * np.sqrt(1 - xi)

        xi = brentq(rayleigh, 0.5, 0.9999)
        return float(self.c_t * np.sqrt(xi))


STEEL = Material("steel", density=7850.0, c_l=5920.0, c_t=3230.0)
STAINLESS_304 = Material("stainless steel 304", density=7900.0, c_l=5790.0, c_t=3100.0)
ALUMINIUM = Material("aluminium", density=2700.0, c_l=6320.0, c_t=3130.0)

MATERIALS = {m.name: m for m in (STEEL, STAINLESS_304, ALUMINIUM)}
