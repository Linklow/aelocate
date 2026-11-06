import numpy as np
import pytest

from aelocate import ChannelHitDetector, HitDetector, HitDetectorConfig, dbae_to_volts, volts_to_dbae

FS = 2e6


def burst(n_total, start, amp, f0=150e3, cycles=12, decay=40e-6):
    t = np.arange(n_total) / FS
    tt = t - start / FS
    x = amp * np.sin(2 * np.pi * f0 * tt) * np.exp(-np.maximum(tt, 0) / decay)
    x[tt < 0] = 0
    x[tt > cycles / f0 * 4] = 0
    return x


def run(x, block=None, cfg=None):
    det = ChannelHitDetector(0, FS, cfg or HitDetectorConfig())
    if block is None:
        hits = det.process(x)
    else:
        hits = []
        for i in range(0, x.size, block):
            hits += det.process(x[i:i + block])
    return hits + det.flush()


def test_single_burst_features():
    amp = float(dbae_to_volts(80))
    x = burst(20000, 3000, amp)
    (h,) = run(x)
    assert h.index == 3000 or h.index == 3001
    assert h.amplitude_db == pytest.approx(80, abs=0.6)
    assert 0 < h.rise_time < 10e-6
    assert h.counts > 3 and h.counts_to_peak <= h.counts
    assert h.energy_aj > 0
    assert h.waveform.size == HitDetectorConfig().waveform_length
    assert h.pretrigger == HitDetectorConfig().pretrigger
    # arrival interpolation lands between the sample before and the first sample above threshold
    assert 2999 <= h.arrival_time * FS <= 3001


def test_below_threshold_gives_no_hits():
    x = burst(20000, 3000, float(dbae_to_volts(40)))
    assert run(x) == []


def test_hdt_splits_separate_bursts_and_hlt_suppresses_echo():
    amp = float(dbae_to_volts(75))
    cfg = HitDetectorConfig(hdt=200e-6, hlt=1000e-6, pdt=50e-6)
    # Two bursts 5 ms apart -> two hits.
    x = burst(40000, 2000, amp) + burst(40000, 12000, amp)
    assert len(run(x, cfg=cfg)) == 2
    # An echo 600 us after the hit ends falls inside HLT -> suppressed.
    y = burst(40000, 2000, amp) + burst(40000, 2000 + 1600, 0.5 * amp)
    hits = run(y, cfg=cfg)
    assert len(hits) == 1


@pytest.mark.parametrize("block", [1, 7, 333, 4096, 10000])
def test_block_size_invariance(block):
    rng = np.random.default_rng(3)
    n = 60000
    x = rng.normal(0, 15e-6, n)
    for start, db in ((1000, 70), (9000, 55), (9600, 60), (30000, 90), (52000, 48)):
        x += burst(n, start, float(dbae_to_volts(db)))
    ref = run(x)
    got = run(x, block=block)
    assert len(ref) == len(got) >= 3
    for a, b in zip(ref, got, strict=True):
        assert (a.index, a.counts, a.counts_to_peak) == (b.index, b.counts, b.counts_to_peak)
        assert a.arrival_time == pytest.approx(b.arrival_time)
        assert a.amplitude_db == pytest.approx(b.amplitude_db)
        assert a.duration == pytest.approx(b.duration)
        assert a.energy_aj == pytest.approx(b.energy_aj)
        assert np.array_equal(a.waveform, b.waveform)


def test_negative_polarity_and_peak_definition():
    amp = float(dbae_to_volts(70))
    x = -burst(20000, 4000, amp)
    (h,) = run(x)
    assert h.amplitude_db == pytest.approx(70, abs=0.6)
    assert volts_to_dbae(np.abs(h.waveform).max()) == pytest.approx(h.amplitude_db, abs=0.01)


def test_multichannel_sorted_by_arrival():
    amp = float(dbae_to_volts(70))
    data = np.stack([burst(20000, 5000, amp), burst(20000, 2000, amp)])
    det = HitDetector(2, FS)
    hits = det.process(data) + det.flush()
    assert [h.channel for h in hits] == [1, 0]
    with pytest.raises(ValueError):
        det.process(np.zeros((3, 10)))
