"""Location-accuracy benchmark: standard delta-T vs AIC picking vs neural-network correction.

The protocol follows ordinary practice on a real structure:

1. **Velocity calibration.** Hsu-Nielsen breaks (PLB, ~95 dB) at marked points between the
   sensors give an effective wave speed for each picker.
2. **Training (ML only).** Events from known points along the specimen (including outside the
   sensor base): pulser "location pulses" with variable gain, plus pencil-lead breaks of varied
   strength. An ablation trains on pulser events only. Its poor transfer to step-like sources
   (domain shift) is one of the findings.
3. **Test.** Crack-like sources (0.6 us rise time, which neither calibration nor training used)
   with amplitudes of 45-85 dB at random positions.

Each method is scored on location error and on the **probability of correct location**: the
share of *all* generated events placed within ``tol`` of the true position, so events that
were missed or not located count as failures.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .daq import NoiseGenerator, simulate_event
from .events import Event, EventBuilder
from .hits import HitDetector
from .location import calibrate_velocity
from .pickers import AICPicker, ThresholdPicker
from .presets import Setup, steel_plate_linear
from .sources import CRACK, PLB, PULSER


@dataclass(frozen=True, eq=False)
class Trial:
    x_true: float
    amplitude_db: float
    source: str
    event: Event | None
    t_offset: float = 0.0
    """Add to hit times to get time relative to the source (t0 = 0)."""


def simulate_trials(setup: Setup, source, positions: Sequence[float], amplitudes: Sequence[float],
                    rng: np.random.Generator) -> list[Trial]:
    """Simulate isolated events and run them through hit detection and event building."""
    sim = setup.simulator
    noise = NoiseGenerator(setup.frontend, setup.n_channels, rng)
    window = setup.event_window()
    trials = []
    for x, amp in zip(positions, amplitudes, strict=True):
        start, data = simulate_event(sim, source, [x], float(amp), noise)
        det = HitDetector(setup.n_channels, setup.fs, setup.detector)
        hits = det.process(data) + det.flush()
        builder = EventBuilder(window)
        builder.push(hits)
        events = builder.pop()
        trials.append(Trial(float(x), float(amp), source.name, events[0] if events else None, start / setup.fs))
    return trials


def locate_trials(trials: Sequence[Trial], locator) -> np.ndarray:
    """Estimated x for each trial (NaN where the event was missed or could not be located)."""
    out = np.full(len(trials), np.nan)
    for i, t in enumerate(trials):
        if t.event is not None:
            loc = locator.locate(t.event)
            if loc is not None:
                out[i] = loc.x
    return out


def zone_masks(x_true: np.ndarray, sensor_x: Sequence[float], near: float = 0.10) -> dict[str, np.ndarray]:
    lo, hi = float(min(sensor_x)), float(max(sensor_x))
    inside = (x_true >= lo) & (x_true <= hi)
    d_sensor = np.min(np.abs(x_true[:, None] - np.asarray(sensor_x)[None, :]), axis=1)
    return {
        "all": np.ones_like(x_true, dtype=bool),
        "inside base": inside,
        "near sensor (<=10 cm, inside)": inside & (d_sensor <= near),
        "mid-span": inside & (d_sensor > near),
        "outside base": ~inside,
    }


def summarize(x_true: np.ndarray, x_est: np.ndarray, mask: np.ndarray, tol: float) -> dict[str, float]:
    xt, xe = x_true[mask], x_est[mask]
    located = ~np.isnan(xe)
    err = np.abs(xe[located] - xt[located])
    n = int(mask.sum())
    return {
        "n": n,
        "located_pct": 100.0 * located.mean() if n else float("nan"),
        "mae_mm": 1e3 * float(err.mean()) if err.size else float("nan"),
        "median_mm": 1e3 * float(np.median(err)) if err.size else float("nan"),
        "p90_mm": 1e3 * float(np.percentile(err, 90)) if err.size else float("nan"),
        "p_correct_pct": 100.0 * float((err <= tol).sum()) / n if n else float("nan"),
    }


@dataclass
class BenchmarkResult:
    setup_name: str
    tol: float
    velocities: dict[str, float]
    metrics: dict[str, dict[str, dict[str, float]]]
    x_true: np.ndarray = field(repr=False)
    amplitude: np.ndarray = field(repr=False)
    estimates: dict[str, np.ndarray] = field(repr=False)
    example: Trial | None = field(default=None, repr=False)

    def table(self, zone: str = "all") -> str:
        head = (f"| Method | Located | MAE, mm | Median, mm | P90, mm | P(correct, ±{self.tol * 1e3:.0f} mm) |\n"
                "|---|---:|---:|---:|---:|---:|\n")
        rows = []
        for method, zones in self.metrics.items():
            m = zones[zone]
            rows.append(f"| {method} | {m['located_pct']:.1f}% | {m['mae_mm']:.1f} | {m['median_mm']:.1f} | "
                        f"{m['p90_mm']:.1f} | {m['p_correct_pct']:.1f}% |")
        return head + "\n".join(rows)

    def to_json(self) -> dict:
        return {"setup": self.setup_name, "tol_m": self.tol, "velocities_m_s": self.velocities,
                "metrics": self.metrics}


METHOD_LABELS = {
    "standard": "Standard ΔT (threshold)",
    "aic": "AIC picker",
    "ml": "Neural network",
    "ml_pulser": "Neural network, pulser-only training (ablation)",
}
PLOTTED_METHODS = ("standard", "aic", "ml")


def _fit(ml, trials: Sequence[Trial]):
    ok = [t for t in trials if t.event is not None]
    return ml.fit([t.event for t in ok], [t.x_true for t in ok])


def calibrate(setup: Setup, rng: np.random.Generator, pickers=None) -> dict[str, float]:
    """Effective velocity per picker from ~95 dB pencil-lead breaks at marked points (2-sensor zone 0-1)."""
    sx = np.asarray(setup.specimen.sensor_x)
    xa, xb = float(sx[0]), float(sx[1])
    pickers = pickers or {"standard": ThresholdPicker(), "aic": AICPicker()}
    cal_x = np.repeat(np.linspace(xa + 0.1 * (xb - xa), xb - 0.1 * (xb - xa), 9), 5)
    cal = simulate_trials(setup, PLB, cal_x, rng.uniform(93, 97, cal_x.size), rng)
    velocities = {}
    for name, picker in pickers.items():
        dts, xs = [], []
        for t in cal:
            h0 = t.event.hit_on(0) if t.event is not None else None
            h1 = t.event.hit_on(1) if t.event is not None else None
            if h0 is not None and h1 is not None:
                dts.append(picker.pick(h0) - picker.pick(h1))
                xs.append(t.x_true)
        velocities[name], _ = calibrate_velocity(dts, xs, xa, xb)
    return velocities


def training_trials(setup: Setup, n: int, rng: np.random.Generator, margin: float = 0.15,
                    pulser_share: float = 0.6) -> tuple[list[Trial], list[Trial]]:
    """Known-position training events: (pulser location pulses, pencil-lead breaks)."""
    sx = np.asarray(setup.specimen.sensor_x)
    lo_x, hi_x = max(0.0, sx[0] - margin), min(setup.specimen.length, sx[-1] + margin)
    n_p = int(round(pulser_share * n))
    n_b = n - n_p
    pulser = simulate_trials(setup, PULSER, rng.uniform(lo_x, hi_x, n_p), rng.uniform(45, 95, n_p), rng)
    plb = simulate_trials(setup, PLB, rng.uniform(lo_x, hi_x, n_b), rng.uniform(45, 95, n_b), rng)
    return pulser, plb


def train_ml_locator(setup: Setup, n: int = 3000, seed: int = 0):
    """Calibrate the velocity and train an :class:`~aelocate.ml.MLLinearLocator` for a set-up."""
    from .ml import MLLinearLocator

    rng = np.random.default_rng(seed)
    velocities = calibrate(setup, rng)
    pulser, plb = training_trials(setup, n, rng)
    ml = _fit(MLLinearLocator(setup.specimen.sensor_x, velocities["standard"], seed=seed), pulser + plb)
    return ml, velocities


def run_benchmark(setup: Setup | None = None, n_train: int = 6000, n_test: int = 3000, tol: float = 0.025,
                  seed: int = 7, verbose: bool = True) -> BenchmarkResult:
    """Run the full calibration -> training -> test protocol and return metrics."""
    from .ml import MLLinearLocator

    setup = setup or steel_plate_linear()
    if not setup.is_linear:
        raise ValueError("the benchmark is defined for linear location set-ups")
    rng = np.random.default_rng(seed)
    sx = np.asarray(setup.specimen.sensor_x)
    log = print if verbose else (lambda *a, **k: None)
    aic = AICPicker()

    # 1. Velocity calibration with pencil-lead breaks at marked points between the sensors.
    velocities = calibrate(setup, rng, {"standard": ThresholdPicker(), "aic": aic})
    log(f"calibrated velocity: threshold {velocities['standard']:.0f} m/s, AIC {velocities['aic']:.0f} m/s")

    # 2. Training set: location pulses + pencil-lead breaks at known points, whole specimen.
    margin = 0.15
    lo_x, hi_x = max(0.0, sx[0] - margin), min(setup.specimen.length, sx[-1] + margin)
    train_pulser, train_plb = training_trials(setup, n_train, rng, margin)
    ml = _fit(MLLinearLocator(sx, velocities["standard"], aic=aic, seed=seed), train_pulser + train_plb)
    ml_pulser = _fit(MLLinearLocator(sx, velocities["standard"], aic=aic, seed=seed), train_pulser)
    log(f"trained MLPs on {len(train_pulser)} pulser + {len(train_plb)} PLB events")

    # 3. Test set: crack-like sources the model has never seen.
    xb = float(sx[-1])
    test_x = rng.uniform(lo_x, hi_x, n_test)
    test_amp = rng.uniform(45, 85, n_test)
    test = simulate_trials(setup, CRACK, test_x, test_amp, rng)

    locators = {
        "standard": setup.locator(velocities["standard"], ThresholdPicker()),
        "aic": setup.locator(velocities["aic"], aic),
        "ml": ml,
        "ml_pulser": ml_pulser,
    }
    x_true = np.array([t.x_true for t in test])
    estimates = {name: locate_trials(test, loc) for name, loc in locators.items()}
    masks = zone_masks(x_true, sx)
    metrics = {METHOD_LABELS[name]: {z: summarize(x_true, est, m, tol) for z, m in masks.items()}
               for name, est in estimates.items()}

    example = next((t for t in test if t.event is not None and len(t.event.hits) == 2
                    and 60 < t.amplitude_db < 70 and xb - 0.25 < t.x_true < xb - 0.1), None)
    return BenchmarkResult(setup.name, tol, velocities, metrics, x_true, np.array(test_amp), estimates, example)


def write_report(result: BenchmarkResult, setup: Setup, out_dir: str | Path) -> list[Path]:
    """Save figures, ``results.md`` and ``results.json`` into ``out_dir``."""
    from . import plots

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = [
        plots.dispersion_figure(setup, out / "dispersion.png"),
        plots.error_vs_position_figure(result, setup, out / "error_vs_position.png"),
        plots.probability_vs_position_figure(result, setup, out / "probability_vs_position.png"),
        plots.error_vs_amplitude_figure(result, out / "error_vs_amplitude.png"),
    ]
    if result.example is not None:
        files.append(plots.example_event_figure(result.example, setup, result.velocities, out / "example_event.png"))
    md = [f"# Location benchmark: {result.setup_name}", "",
          f"Calibrated velocity: threshold {result.velocities['standard']:.0f} m/s, "
          f"AIC {result.velocities['aic']:.0f} m/s.", ""]
    for zone in ("all", "inside base", "near sensor (<=10 cm, inside)", "mid-span", "outside base"):
        md += [f"## {zone}", "", result.table(zone), ""]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return [*files, out / "results.md", out / "results.json"]
