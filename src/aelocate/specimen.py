"""Test objects (specimens) and sensor layouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import STEEL, Material


@dataclass(frozen=True)
class _Plate:
    """Wave-propagation properties shared by all plate-like specimens.

    Attributes:
        material: Plate material.
        thickness: Plate thickness, m.
        attenuation_db_per_m: Material + leakage attenuation at ``attenuation_f_ref``, dB/m.
            It scales linearly with frequency.
        spreading_exponent: Geometric spreading, amplitude ~ r^-n. 0.5 = cylindrical spreading
            in a plate; 0 = a narrow strip or pipe acting as a 1-D waveguide.
        s0_ratio: Out-of-plane amplitude of S0 relative to A0 for a surface source.
    """

    material: Material = STEEL
    thickness: float = 5e-3
    attenuation_db_per_m: float = 3.0
    attenuation_f_ref: float = 150e3
    spreading_exponent: float = 0.5
    s0_ratio: float = 0.12


@dataclass(frozen=True)
class LinearSpecimen(_Plate):
    """A long object with sensors on a line: strip, pipe, rail, weld seam.

    Attributes:
        length: Object length, m. The ends can reflect waves (image-source model).
        sensor_x: Sensor coordinates along the object, m, ascending. Channel ``i`` is ``sensor_x[i]``.
        reflection_coefficient: Amplitude reflection coefficient of the ends.
        reflection_orders: Number of reflections to model (0 disables reflections).
    """

    length: float = 1.5
    sensor_x: tuple[float, ...] = (0.25, 1.25)
    reflection_coefficient: float = 0.5
    reflection_orders: int = 0
    kind: str = field(default="linear", init=False)

    def __post_init__(self) -> None:
        xs = np.asarray(self.sensor_x, dtype=float)
        if xs.size < 2 or np.any(np.diff(xs) <= 0):
            raise ValueError("sensor_x must contain at least two strictly ascending coordinates")
        if xs[0] < 0 or xs[-1] > self.length:
            raise ValueError("sensors must lie on the specimen")

    @property
    def n_channels(self) -> int:
        return len(self.sensor_x)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, self.length),)

    def clamp(self, position) -> np.ndarray:
        return np.clip(np.atleast_1d(np.asarray(position, dtype=float))[:1], 0.0, self.length)

    def paths(self, position, channel: int) -> list[tuple[float, float]]:
        """Propagation paths (distance, amplitude factor) from a source to a sensor."""
        xs = float(np.atleast_1d(position)[0])
        xr = self.sensor_x[channel]
        out = [(abs(xs - xr), 1.0)]
        n_max = self.reflection_orders
        if n_max > 0 and self.reflection_coefficient != 0:
            for m in range(-n_max, n_max + 1):
                for img, n_refl in ((2 * m * self.length + xs, abs(2 * m)),
                                    (2 * m * self.length - xs, abs(2 * m - 1))):
                    if 1 <= n_refl <= n_max:
                        out.append((abs(img - xr), self.reflection_coefficient**n_refl))
        return out


@dataclass(frozen=True)
class PlanarSpecimen(_Plate):
    """A plate with sensors on its surface (vessel wall, tank floor, wing skin).

    Attributes:
        width: Plate size along x, m.
        height: Plate size along y, m.
        sensor_xy: Sensor coordinates ((x0, y0), (x1, y1), ...), m.
    """

    width: float = 1.0
    height: float = 1.0
    sensor_xy: tuple[tuple[float, float], ...] = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    kind: str = field(default="planar", init=False)

    def __post_init__(self) -> None:
        if len(self.sensor_xy) < 3:
            raise ValueError("planar location needs at least three sensors")

    @property
    def n_channels(self) -> int:
        return len(self.sensor_xy)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, self.width), (0.0, self.height))

    def clamp(self, position) -> np.ndarray:
        p = np.asarray(position, dtype=float)[:2]
        return np.clip(p, [0.0, 0.0], [self.width, self.height])

    def paths(self, position, channel: int) -> list[tuple[float, float]]:
        p = np.asarray(position, dtype=float)[:2]
        return [(float(np.hypot(*(p - np.asarray(self.sensor_xy[channel])))), 1.0)]


Specimen = LinearSpecimen | PlanarSpecimen
