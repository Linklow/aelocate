"""Arrival-time pickers.

The threshold crossing is what every AE instrument reports, but it is biased: a weak signal
crosses the threshold later in its rise than a strong one, so the time difference between two
sensors (and the location) depends on amplitude. The Akaike Information Criterion (AIC) picker
(Maeda, 1985; widely used in AE after Kurz et al., 2005) finds the onset as the point that best
splits the window into "noise" and "signal" segments, and is largely amplitude independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .hits import Hit


def aic_curve(x: np.ndarray, min_segment: int = 8) -> np.ndarray:
    """Maeda's AIC function, ``AIC(k) = k log var(x[:k]) + (N-k-1) log var(x[k:])``.

    Computed in O(N) with cumulative sums. Values within ``min_segment`` of either edge are
    ``+inf``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    aic = np.full(n, np.inf)
    if n < 2 * min_segment + 1:
        return aic
    x = x - x.mean()
    c1 = np.cumsum(x)
    c2 = np.cumsum(x * x)
    k = np.arange(min_segment, n - min_segment)
    n_l, n_r = k, n - k
    s_l, s2_l = c1[k - 1], c2[k - 1]
    s_r, s2_r = c1[-1] - s_l, c2[-1] - s2_l
    var_l = np.maximum(s2_l / n_l - (s_l / n_l) ** 2, 0.0)
    var_r = np.maximum(s2_r / n_r - (s_r / n_r) ** 2, 0.0)
    eps = 1e-12 * (c2[-1] / n + 1e-300)
    aic[k] = n_l * np.log(var_l + eps) + (n_r - 1) * np.log(var_r + eps)
    return aic


def aic_onset(x: np.ndarray, min_segment: int = 8) -> int:
    """Index of the AIC minimum (estimated signal onset) in ``x``."""
    return int(np.argmin(aic_curve(x, min_segment)))


class Picker(Protocol):
    name: str

    def pick(self, hit: Hit) -> float:
        """Arrival time of the hit, s."""
        ...


@dataclass(frozen=True)
class ThresholdPicker:
    """First threshold crossing, as reported by the hit detector."""

    name: str = "threshold"

    def pick(self, hit: Hit) -> float:
        return hit.arrival_time


@dataclass(frozen=True)
class AICPicker:
    """AIC onset searched in a window around the first threshold crossing.

    Attributes:
        before: Window start, seconds before the threshold crossing.
        after: Window end, seconds after the threshold crossing.
    """

    before: float = 60e-6
    after: float = 15e-6
    name: str = "aic"

    def pick(self, hit: Hit) -> float:
        p = hit.pretrigger
        a = max(0, p - int(round(self.before * hit.fs)))
        b = min(hit.waveform.size, p + int(round(self.after * hit.fs)) + 1)
        onset = aic_onset(hit.waveform[a:b])
        return hit.time_of_sample(a + onset)
