import numpy as np
import pytest

from aelocate import STEEL, dbae_to_volts, lamb_dispersion, steel_plate_linear, volts_to_dbae
from aelocate.dispersion import kirchhoff_a0_phase_velocity
from aelocate.instrument import FrontEnd
from aelocate.sources import CRACK, PLB


def test_dbae_round_trip():
    assert volts_to_dbae(1e-6) == pytest.approx(0.0)
    assert volts_to_dbae(0.1) == pytest.approx(100.0)
    assert dbae_to_volts(45) == pytest.approx(177.8e-6, rel=1e-3)
    db = np.array([20.0, 45.0, 99.0])
    assert np.allclose(volts_to_dbae(dbae_to_volts(db)), db)


def test_frontend_clips_at_100_db_for_40_db_gain():
    fe = FrontEnd()
    assert volts_to_dbae(fe.clip_level) == pytest.approx(100.0)
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(fe.digitize(x), [-fe.clip_level, 0, fe.clip_level], atol=fe.lsb)


def test_material_velocities():
    assert STEEL.poisson == pytest.approx(0.288, abs=2e-3)
    assert STEEL.c_plate == pytest.approx(5414, abs=2)
    # Viktorov approximation, accurate to ~0.5 %
    nu = STEEL.poisson
    assert STEEL.c_rayleigh == pytest.approx(STEEL.c_t * (0.862 + 1.14 * nu) / (1 + nu), rel=5e-3)


@pytest.fixture(scope="module")
def curves():
    return lamb_dispersion(STEEL, 5e-3, f_max=2.2e6)


def test_s0_low_frequency_limit_is_plate_velocity(curves):
    assert curves["S0"].phase_at(10e3) == pytest.approx(STEEL.c_plate, rel=2e-3)


def test_a0_low_frequency_matches_kirchhoff(curves):
    f = 10e3
    c_k = kirchhoff_a0_phase_velocity(STEEL, 5e-3, f)
    assert curves["A0"].phase_at(f) == pytest.approx(c_k, rel=0.03)
    assert curves["A0"].phase_at(f) < c_k  # Kirchhoff theory over-predicts
    assert curves["A0"].group_at(f) == pytest.approx(2 * curves["A0"].phase_at(f), rel=0.05)


def test_both_modes_approach_rayleigh_at_high_frequency(curves):
    f = 2.0e6  # f*d = 10 MHz*mm
    for mode in ("A0", "S0"):
        assert curves[mode].phase_at(f) == pytest.approx(STEEL.c_rayleigh, rel=0.01)


def test_a0_slower_than_s0_below_s0_dispersion(curves):
    # Below f*d ~ 1.25 MHz*mm S0 is nearly non-dispersive and ~1.7x faster than A0.
    f = np.linspace(100e3, 250e3, 20)
    assert np.all(curves["A0"].group_at(f) < 0.65 * curves["S0"].group_at(f))
    # Around f*d = 2.5 MHz*mm the S0 group velocity drops sharply (to ~1.8 km/s in steel).
    assert curves["S0"].group_at(500e3) < 2000


def test_source_spectra_are_finite_and_broadband():
    f = np.linspace(0, 1e6, 513)
    for src in (PLB, CRACK):
        s = src.spectrum(f)
        assert s[0] == 0 and np.all(np.isfinite(s))
    # a shorter rise time keeps more high-frequency content
    hi = f > 600e3
    assert np.abs(CRACK.spectrum(f))[hi].sum() > np.abs(PLB.spectrum(f))[hi].sum()


def test_simulated_arrivals_follow_mode_velocities():
    setup = steel_plate_linear()
    sim = setup.simulator
    x = 0.45  # 0.2 m from sensor 1, 0.8 m from sensor 2
    start, w = sim.waveforms(PLB, [x], 90.0, t0=0.0)
    env = np.abs(w)
    t = (start + np.arange(w.shape[1])) / sim.fs
    t_a0 = sim.arrival_times([x], mode="A0")
    # the A0 wave packet (most of the energy) peaks shortly after the A0 group arrival
    for ch in range(2):
        t_peak = t[np.argmax(env[ch])]
        assert t_a0[ch] - 10e-6 < t_peak < t_a0[ch] + 80e-6
    # amplitude decreases with distance
    assert env[0].max() > env[1].max()


def test_reference_amplitude_definition():
    setup = steel_plate_linear()
    sim = setup.simulator
    # sensor 1 at x=0.25 m; a source 0.1 m away must peak at the requested dB_AE
    _, w = sim.waveforms(PLB, [0.35], 70.0, t0=0.0)
    assert volts_to_dbae(np.abs(w[0]).max()) == pytest.approx(70.0, abs=0.3)
