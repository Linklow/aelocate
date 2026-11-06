import numpy as np
import pytest

from aelocate import (
    AICPicker,
    Event,
    LinearLocator,
    PlanarLocator,
    aic_onset,
    calibrate_velocity,
    linear_location,
)
from aelocate.hits import Hit


def fake_hit(channel, t, fs=2e6):
    return Hit(channel, int(t * fs), t, 60.0, 1e-5, 1e-4, 5, 2, 100.0, np.zeros(16, np.float32), 0, fs)


def test_linear_location_exact_and_clamped():
    v = 3000.0
    xa, xb = 0.0, 1.0
    for x in (0.1, 0.5, 0.93):
        x_est, inside = linear_location(x / v, (xb - x) / v, xa, xb, v)
        assert x_est == pytest.approx(x)
        assert inside
    # behind sensor A: |dt| = D/v regardless of how far behind -> clamped to the sensor
    x_est, inside = linear_location(0.0, 1.0 / v + 1e-5, xa, xb, v)
    assert x_est == 0.0 and not inside


def test_calibrate_velocity_recovers_speed():
    v = 5300.0
    xs = np.linspace(0.35, 1.15, 9)
    dt = (2 * xs - 0.25 - 1.25) / v
    est, resid = calibrate_velocity(dt, xs, 0.25, 1.25)
    assert est == pytest.approx(v)
    assert resid < 1e-9


def test_linear_locator_picks_adjacent_zone():
    loc = LinearLocator([0.0, 2.0, 4.0, 6.0], 3000.0)
    x_true = 2.6
    hits = [fake_hit(ch, abs(x_true - s) / 3000.0) for ch, s in enumerate([0.0, 2.0, 4.0, 6.0])]
    ev = Event(tuple(sorted(hits, key=lambda h: h.arrival_time)))
    res = loc.locate(ev)
    assert res.channels == (1, 2)
    assert res.x == pytest.approx(x_true)


def test_linear_locator_needs_two_hits():
    loc = LinearLocator([0.0, 1.0], 3000.0)
    assert loc.locate(Event((fake_hit(0, 0.0),))) is None


def test_planar_location_recovers_source():
    sensors = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    v = 5300.0
    loc = PlanarLocator(sensors, v, ((0, 1), (0, 1)))
    for p in ((0.3, 0.7), (0.5, 0.5), (0.85, 0.2)):
        t = [np.hypot(p[0] - s[0], p[1] - s[1]) / v + 1e-3 for s in sensors]
        x, y, t0, rms = loc.solve(range(4), t)
        assert (x, y) == pytest.approx(p, abs=1e-4)
        assert t0 == pytest.approx(1e-3, abs=1e-7)


def test_aic_finds_variance_change():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 8, 200)])
    assert abs(aic_onset(x) - 300) <= 3


def test_aic_picker_uses_window_around_crossing():
    fs = 2e6
    rng = np.random.default_rng(1)
    wf = rng.normal(0, 1e-5, 2048)
    onset = 480
    wf[onset:] += 2e-4 * np.sin(2 * np.pi * 150e3 * np.arange(2048 - onset) / fs)
    h = Hit(0, 512 + 1000, (512 + 1000) / fs, 60, 1e-5, 1e-4, 5, 2, 10, wf.astype(np.float32), 1000, fs)
    t = AICPicker().pick(h)
    assert abs(t * fs - (1000 + onset)) <= 4
