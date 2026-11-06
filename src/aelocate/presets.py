"""Ready-made measurement set-ups."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from .hits import HitDetectorConfig
from .instrument import FrontEnd, Sensor
from .location import LinearLocator, PlanarLocator
from .pickers import Picker
from .simulator import WaveformSimulator
from .specimen import LinearSpecimen, PlanarSpecimen, Specimen


@dataclass(frozen=True)
class Setup:
    """Specimen + sensors + acquisition settings: everything needed to simulate and process."""

    name: str
    specimen: Specimen
    sensor: Sensor = field(default_factory=Sensor)
    frontend: FrontEnd = field(default_factory=FrontEnd)
    detector: HitDetectorConfig = field(default_factory=HitDetectorConfig)

    @cached_property
    def simulator(self) -> WaveformSimulator:
        return WaveformSimulator(self.specimen, self.sensor, self.frontend)

    @property
    def fs(self) -> float:
        return self.frontend.fs

    @property
    def n_channels(self) -> int:
        return self.specimen.n_channels

    @property
    def is_linear(self) -> bool:
        return isinstance(self.specimen, LinearSpecimen)

    def group_velocity(self, mode: str = "A0", f: float | None = None) -> float:
        """Group velocity of a mode at ``f`` (default: sensor resonance), m/s."""
        f = self.sensor.f_res if f is None else f
        return float(self.simulator.curves[mode].group_at(f))

    def sensor_positions(self) -> np.ndarray:
        if isinstance(self.specimen, LinearSpecimen):
            return np.asarray(self.specimen.sensor_x, dtype=float)
        return np.asarray(self.specimen.sensor_xy, dtype=float)

    def event_window(self, margin: float = 1.25) -> float:
        """Event definition time: longest relevant sensor-to-sensor travel time x margin."""
        pos = self.sensor_positions()
        if pos.ndim == 1:
            spacing = float(np.max(np.diff(pos)))
        else:
            spacing = float(np.max(np.linalg.norm(pos[:, None] - pos[None], axis=-1)))
        slowest = self.group_velocity("A0", self.frontend.band[0])
        return margin * spacing / slowest

    def locator(self, velocity: float, picker: Picker | None = None):
        if isinstance(self.specimen, LinearSpecimen):
            return LinearLocator(self.specimen.sensor_x, velocity, picker)
        return PlanarLocator(self.specimen.sensor_xy, velocity, self.specimen.bounds, picker)

    def with_detector(self, **changes) -> Setup:
        return replace(self, detector=replace(self.detector, **changes))


def steel_plate_linear(reflections: int = 0) -> Setup:
    """Linear location on a 5 mm steel plate, two 150 kHz sensors 1 m apart.

    This mirrors a classic laboratory set-up for studying linear-location error: a flat steel
    plate, sensors on a line, sources anywhere along that line, inside and outside the sensor base.
    """
    specimen = LinearSpecimen(length=1.5, sensor_x=(0.25, 1.25), thickness=5e-3,
                              attenuation_db_per_m=3.0, spreading_exponent=0.5, s0_ratio=0.12,
                              reflection_orders=reflections, reflection_coefficient=0.4)
    return Setup("steel plate, linear location (2 sensors, 1 m base)", specimen)


def steel_pipe_linear() -> Setup:
    """A 6 m pipe section with four sensors 2 m apart (waveguide: little geometric spreading)."""
    specimen = LinearSpecimen(length=6.0, sensor_x=(0.0, 2.0, 4.0, 6.0), thickness=8e-3,
                              attenuation_db_per_m=4.0, spreading_exponent=0.1, s0_ratio=0.1)
    return Setup("steel pipe, linear location (4 sensors, 2 m spacing)", specimen,
                 detector=HitDetectorConfig(threshold_db=42.0))


def steel_plate_planar() -> Setup:
    """Planar location on a 1 x 1 m, 5 mm steel plate with four sensors near the corners."""
    specimen = PlanarSpecimen(width=1.0, height=1.0, thickness=5e-3,
                              sensor_xy=((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
    return Setup("steel plate, planar location (4 sensors)", specimen)


PRESETS = {
    "plate-linear": steel_plate_linear,
    "pipe-linear": steel_pipe_linear,
    "plate-planar": steel_plate_planar,
}
