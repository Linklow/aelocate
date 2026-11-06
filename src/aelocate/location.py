"""AE source location: linear (1-D, delta-T method) and planar (2-D, TDOA least squares)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .events import Event
from .pickers import Picker, ThresholdPicker


@dataclass(frozen=True, eq=False)
class LocatedEvent:
    """Result of locating one event."""

    x: float
    y: float | None
    inside: bool
    """For linear location: |dT| <= D/v, i.e. the source was placed between the two sensors."""
    channels: tuple[int, ...]
    dt: float
    """Time difference used for location (t_lower_sensor - t_upper_sensor for linear), s."""
    method: str
    event: Event
    residual: float = 0.0

    @property
    def time(self) -> float:
        return self.event.time

    @property
    def position(self) -> tuple[float, ...]:
        return (self.x,) if self.y is None else (self.x, self.y)


def linear_location(t_a: float, t_b: float, x_a: float, x_b: float, velocity: float) -> tuple[float, bool]:
    """Standard two-sensor linear location.

    With sensors at ``x_a < x_b`` and arrival times ``t_a, t_b``:

        x = (x_a + x_b) / 2 + v * (t_a - t_b) / 2

    If ``|t_a - t_b| > (x_b - x_a) / v`` the source cannot be between the sensors. It is
    clamped to the nearer sensor and ``inside`` is False. Real instruments do the same, and
    sources near or behind a sensor pile up at the sensor position because of it.

    Returns:
        ``(x, inside)``.
    """
    if x_b <= x_a:
        raise ValueError("expected x_a < x_b")
    dt = t_a - t_b
    x = 0.5 * (x_a + x_b) + 0.5 * velocity * dt
    inside = abs(dt) * velocity <= (x_b - x_a)
    return float(np.clip(x, x_a, x_b)), bool(inside)


def calibrate_velocity(dt: Sequence[float], x_true: Sequence[float], x_a: float, x_b: float) -> tuple[float, float]:
    """Effective wave speed from sources at known positions (e.g. Hsu-Nielsen breaks).

    Least-squares fit of ``2x - x_a - x_b = v * dt`` through the origin.

    Returns:
        ``(velocity, rms_position_residual)``. The residual is in metres.
    """
    dt = np.asarray(dt, dtype=float)
    y = 2 * np.asarray(x_true, dtype=float) - x_a - x_b
    v = float(np.dot(dt, y) / np.dot(dt, dt))
    resid = 0.5 * (y - v * dt)
    return v, float(np.sqrt(np.mean(resid**2)))


class LinearLocator:
    """Linear location along a line of sensors (two or more).

    The first-hit sensor and its adjacent sensor with the earliest hit define the zone; the
    source is located between those two sensors.

    Args:
        sensor_x: Sensor coordinates, ascending; channel ``i`` is at ``sensor_x[i]``.
        velocity: Wave speed used for location, m/s.
        picker: Arrival-time picker.
    """

    def __init__(self, sensor_x: Sequence[float], velocity: float, picker: Picker | None = None) -> None:
        self.sensor_x = np.asarray(sensor_x, dtype=float)
        if np.any(np.diff(self.sensor_x) <= 0):
            raise ValueError("sensor_x must be strictly ascending")
        self.velocity = float(velocity)
        self.picker = picker or ThresholdPicker()

    @property
    def method(self) -> str:
        return self.picker.name

    def zone(self, event: Event) -> tuple[int, int] | None:
        """(lower_channel, upper_channel) of the location zone, or None if not locatable."""
        if len(event.hits) < 2:
            return None
        i = event.first.channel
        neighbours = [event.hit_on(j) for j in (i - 1, i + 1) if 0 <= j < self.sensor_x.size]
        neighbours = [h for h in neighbours if h is not None]
        if not neighbours:
            return None
        j = min(neighbours, key=lambda h: h.arrival_time).channel
        return (i, j) if i < j else (j, i)

    def locate(self, event: Event) -> LocatedEvent | None:
        zone = self.zone(event)
        if zone is None:
            return None
        a, b = zone
        ha, hb = event.hit_on(a), event.hit_on(b)
        assert ha is not None and hb is not None
        t_a, t_b = self.picker.pick(ha), self.picker.pick(hb)
        x, inside = linear_location(t_a, t_b, self.sensor_x[a], self.sensor_x[b], self.velocity)
        return LocatedEvent(x, None, inside, (a, b), t_a - t_b, self.method, event)


class PlanarLocator:
    """Planar location by least-squares time-difference-of-arrival (TDOA).

    Solves for source ``(x, y)`` and origin time ``t0`` minimising
    ``sum_i (t_i - t0 - |p - s_i| / v)^2`` over all hit sensors (at least three).
    """

    def __init__(self, sensor_xy: Sequence[Sequence[float]], velocity: float,
                 bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
                 picker: Picker | None = None) -> None:
        self.sensor_xy = np.asarray(sensor_xy, dtype=float)
        self.velocity = float(velocity)
        self.bounds = bounds
        self.picker = picker or ThresholdPicker()

    @property
    def method(self) -> str:
        return self.picker.name

    def solve(self, channels: Sequence[int], times: Sequence[float]) -> tuple[float, float, float, float]:
        """Return ``(x, y, t0, rms_time_residual)`` for arrivals on the given channels."""
        s = self.sensor_xy[list(channels)]
        t = np.asarray(times, dtype=float)
        v = self.velocity
        t_ref = t.min()
        w = np.exp(-(t - t_ref) / (np.ptp(t) + 1e-9))
        p0 = (s * w[:, None]).sum(axis=0) / w.sum()
        x0 = np.array([p0[0], p0[1], t_ref - np.linalg.norm(s[np.argmin(t)] - p0) / v])

        def residuals(q):
            return (t - q[2] - np.hypot(s[:, 0] - q[0], s[:, 1] - q[1]) / v) * v  # metres

        lb, ub = [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]
        if self.bounds is not None:
            (x_lo, x_hi), (y_lo, y_hi) = self.bounds
            lb[:2], ub[:2] = [x_lo, y_lo], [x_hi, y_hi]
            x0[0], x0[1] = np.clip(x0[0], x_lo, x_hi), np.clip(x0[1], y_lo, y_hi)
        sol = least_squares(residuals, x0, bounds=(lb, ub), x_scale=[0.1, 0.1, 1e-4])
        rms = float(np.sqrt(np.mean((sol.fun / v) ** 2)))
        return float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), rms

    def locate(self, event: Event) -> LocatedEvent | None:
        if len(event.hits) < 3:
            return None
        chans = [h.channel for h in event.hits]
        times = [self.picker.pick(h) for h in event.hits]
        x, y, _, rms = self.solve(chans, times)
        return LocatedEvent(x, y, True, tuple(chans), float(times[-1] - times[0]), self.method, event, rms)
