import numpy as np
import pytest

from aelocate import (
    AcquisitionSimulator,
    Emitter,
    ProcessingChain,
    steel_plate_linear,
    steel_plate_planar,
)
from aelocate.benchmark import run_benchmark
from aelocate.sources import PLB


def test_stream_locates_strong_breaks_accurately():
    setup = steel_plate_linear()
    v = setup.group_velocity("S0")
    daq = AcquisitionSimulator(setup.simulator, block_size=20000, seed=5)
    chain = ProcessingChain.from_setup(setup, setup.locator(v))
    truth = []
    for i, x in enumerate((0.4, 0.6, 0.8, 1.0, 1.1)):
        truth.append(daq.inject(PLB, [x], 95.0, t0=0.05 + 0.05 * i))
    located = []
    while daq.time < 0.5:
        located += chain.process(daq.next_block()).located
    located += chain.flush().located
    assert len(located) == len(truth)
    for loc, ev in zip(sorted(located, key=lambda l: l.time), truth, strict=True):
        assert loc.x == pytest.approx(ev.position[0], abs=0.02)  # S0 caught at 95 dB


def test_stream_ground_truth_and_rates():
    setup = steel_plate_linear()
    em = [Emitter("crack", PLB, 20.0, (0.7,), spread=0.0, amplitude_db=(80, 80))]
    daq = AcquisitionSimulator(setup.simulator, em, block_size=40000, seed=2)
    blocks = [daq.next_block() for _ in range(50)]  # 1 s
    n = sum(len(b.events) for b in blocks)
    assert 8 <= n <= 35
    assert all(b.data.dtype == np.float32 and b.data.shape == (2, 40000) for b in blocks)
    assert blocks[1].start == 40000


def test_planar_stream():
    setup = steel_plate_planar()
    v = setup.group_velocity("S0")
    daq = AcquisitionSimulator(setup.simulator, block_size=20000, seed=3)
    chain = ProcessingChain.from_setup(setup, setup.locator(v))
    daq.inject(PLB, [0.3, 0.6], 95.0, t0=0.02)
    located = []
    while daq.time < 0.15:
        located += chain.process(daq.next_block()).located
    located += chain.flush().located
    assert len(located) == 1
    assert (located[0].x, located[0].y) == pytest.approx((0.3, 0.6), abs=0.03)


@pytest.mark.slow
def test_benchmark_ml_beats_standard():
    r = run_benchmark(n_train=1500, n_test=400, verbose=False)
    std = r.metrics["Standard ΔT (threshold)"]["all"]
    ml = r.metrics["Neural network"]["all"]
    assert ml["mae_mm"] < 0.5 * std["mae_mm"]
    assert ml["p_correct_pct"] > std["p_correct_pct"]
