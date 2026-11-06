"""Neural-network correction of linear location.

The standard delta-T algorithm assumes one wave speed and one well-defined arrival. On a plate,
the threshold catches S0 on strong or near signals and A0 on weak or far ones, and the crossing
moves later in the rise as amplitude drops. The resulting error has an additive part (timing
bias) and a multiplicative part (wrong velocity), and both depend on amplitude and position.

:class:`MLLinearLocator` learns the mapping from classic hit features of the two zone sensors
(threshold and AIC time differences, amplitudes, rise times, counts, energies, durations) to
the source coordinate. It is trained on events from known positions, for example pulser
"location pulses" injected at marked points, and then applied to real events. Coordinates are
normalised to the location zone, so a model trained on one sensor pair transfers to another
pair with the same spacing.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .events import Event
from .location import LinearLocator, LocatedEvent, linear_location
from .pickers import AICPicker

FEATURE_NAMES = (
    "dt_thr", "dt_aic", "amp_a", "amp_b", "amp_diff",
    "rise_a", "rise_b", "ctp_a", "ctp_b", "counts_a", "counts_b",
    "dur_a", "dur_b", "log_energy_a", "log_energy_b", "u_thr", "u_aic", "first_is_a",
    "lead_a", "lead_b", "pre_db_a", "pre_db_b",
)


def _precursor(hit, wide: AICPicker) -> tuple[float, float]:
    """S0 precursor of a hit: (lead time of the wide-window AIC onset, pre-trigger RMS in dB_AE).

    When S0 is above the noise but below the threshold, a wide AIC window finds its onset ahead
    of the threshold crossing. The lead grows with source distance (S0 outruns A0), which gives
    the model range information that a single time difference does not carry.
    """
    t_wide = wide.pick(hit)
    p = hit.pretrigger
    n = int(round(wide.before * hit.fs))
    seg = hit.waveform[max(0, p - n):p].astype(float)
    rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    return hit.arrival_time - t_wide, 20 * np.log10(max(rms, 1e-9) / 1e-6)


class MLLinearLocator:
    """Linear location with an MLP that corrects the standard algorithm.

    Args:
        sensor_x: Sensor coordinates, ascending.
        velocity: Wave speed for the standard estimate used as an input feature, m/s.
        aic: AIC picker for the AIC-based features.
        hidden: Hidden-layer sizes of the MLP.
        seed: Random seed for training.
    """

    method = "ml"

    def __init__(self, sensor_x: Sequence[float], velocity: float, aic: AICPicker | None = None,
                 hidden: tuple[int, ...] = (64, 64), seed: int = 0) -> None:
        self._zone_finder = LinearLocator(sensor_x, velocity)
        self.sensor_x = self._zone_finder.sensor_x
        self.velocity = float(velocity)
        self.aic = aic or AICPicker()
        self.wide_aic = AICPicker(before=240e-6, after=5e-6, name="aic-wide")
        self.hidden = hidden
        self.seed = seed
        self.model = None

    def features(self, event: Event) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Feature vector of an event and its location zone, or None if it cannot be located."""
        zone = self._zone_finder.zone(event)
        if zone is None:
            return None
        a, b = zone
        ha, hb = event.hit_on(a), event.hit_on(b)
        assert ha is not None and hb is not None
        xa, xb = self.sensor_x[a], self.sensor_x[b]
        d = xb - xa
        t_ta, t_tb = ha.arrival_time, hb.arrival_time
        t_aa, t_ab = self.aic.pick(ha), self.aic.pick(hb)
        x_thr, _ = linear_location(t_ta, t_tb, xa, xb, self.velocity)
        x_aic, _ = linear_location(t_aa, t_ab, xa, xb, self.velocity)
        tn = d / self.velocity  # time scale of the zone
        lead_a, pre_a = _precursor(ha, self.wide_aic)
        lead_b, pre_b = _precursor(hb, self.wide_aic)
        f = np.array([
            (t_ta - t_tb) / tn, (t_aa - t_ab) / tn,
            ha.amplitude_db / 100, hb.amplitude_db / 100, (ha.amplitude_db - hb.amplitude_db) / 20,
            ha.rise_time / tn, hb.rise_time / tn, np.log1p(ha.counts_to_peak), np.log1p(hb.counts_to_peak),
            np.log1p(ha.counts), np.log1p(hb.counts),
            np.log1p(ha.duration * 1e6), np.log1p(hb.duration * 1e6),
            np.log10(ha.energy_aj + 1), np.log10(hb.energy_aj + 1),
            (x_thr - xa) / d, (x_aic - xa) / d, float(event.first.channel == a),
            lead_a / tn, lead_b / tn, pre_a / 100, pre_b / 100,
        ])
        return f, zone

    def _matrix(self, events: Sequence[Event]):
        rows, zones, keep = [], [], []
        for i, ev in enumerate(events):
            r = self.features(ev)
            if r is not None:
                rows.append(r[0])
                zones.append(r[1])
                keep.append(i)
        return np.array(rows), zones, np.array(keep, dtype=int)

    def fit(self, events: Sequence[Event], x_true: Sequence[float]) -> MLLinearLocator:
        """Train on events whose true source coordinates are known."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, zones, keep = self._matrix(events)
        if keep.size < 20:
            raise ValueError("not enough locatable training events")
        x_true = np.asarray(x_true, dtype=float)[keep]
        xa = np.array([self.sensor_x[a] for a, _ in zones])
        xb = np.array([self.sensor_x[b] for _, b in zones])
        u = (x_true - xa) / (xb - xa)
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=3000, early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=60, random_state=self.seed),
        )
        self.model.fit(X, u)
        return self

    def locate(self, event: Event) -> LocatedEvent | None:
        if self.model is None:
            raise RuntimeError("model is not trained; call fit() or load()")
        r = self.features(event)
        if r is None:
            return None
        f, (a, b) = r
        xa, xb = self.sensor_x[a], self.sensor_x[b]
        u = float(self.model.predict(f[None, :])[0])
        x = xa + u * (xb - xa)
        lo, hi = 0.0, float(self.sensor_x[-1] + (self.sensor_x[-1] - self.sensor_x[0]))
        return LocatedEvent(float(np.clip(x, lo, hi)), None, 0.0 <= u <= 1.0, (a, b),
                            event.hit_on(a).arrival_time - event.hit_on(b).arrival_time, self.method, event)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> MLLinearLocator:
        """Load a model saved with :meth:`save`. Only load files you created yourself (pickle)."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, MLLinearLocator):
            raise TypeError("file does not contain an MLLinearLocator")
        return obj
