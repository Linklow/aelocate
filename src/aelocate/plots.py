"""Static report figures (matplotlib). Requires the ``plots`` extra."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .units import dbae_to_volts  # noqa: E402

# Validated categorical slots (first three pass all-pairs CVD checks) and neutral ink.
SERIES = {"standard": "#2a78d6", "aic": "#eb6834", "ml": "#1baf7a"}
MODE_COLORS = {"A0": "#2a78d6", "S0": "#eb6834"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABELS = {"standard": "Standard ΔT (threshold)", "aic": "AIC picker", "ml": "Neural network"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10, "lines.linewidth": 2.0, "font.family": "sans-serif",
})


def _save(fig, path) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _sensor_lines(ax, sensor_x, vertical: bool = True) -> None:
    for i, x in enumerate(sensor_x):
        (ax.axvline if vertical else ax.axhline)(x, color=MUTED, lw=1, ls="--", zorder=0)
        if vertical:
            ax.annotate(f"S{i + 1}", (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", color=INK2, fontsize=8)


def dispersion_figure(setup, path) -> Path:
    """Phase and group velocity of A0 / S0 with the acquisition band shaded."""
    curves = setup.simulator.curves
    f = np.linspace(10e3, 1.0e6, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    lo, hi = setup.frontend.band
    ax.axvspan(lo / 1e3, hi / 1e3, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.annotate("acquisition band", ((lo + hi) / 2e3, 0.02), xycoords=("data", "axes fraction"),
                ha="center", color=INK2, fontsize=8)
    for mode in ("S0", "A0"):
        c = curves[mode]
        ax.plot(f / 1e3, c.group_at(f), color=MODE_COLORS[mode], label=f"{mode} group velocity")
        ax.plot(f / 1e3, c.phase_at(f), color=MODE_COLORS[mode], ls="--", lw=1.5, label=f"{mode} phase velocity")
    m = setup.specimen.material
    ax.set(xlabel="Frequency, kHz", ylabel="Velocity, m/s", xlim=(0, 1000), ylim=(0, 6000),
           title=f"Lamb-wave dispersion, {m.name}, {setup.specimen.thickness * 1e3:.0f} mm "
                 "(Rayleigh-Lamb solution)")
    ax.legend(loc="lower right", ncols=2)
    return _save(fig, path)


def example_event_figure(trial, setup, velocities, path) -> Path:
    """Both channels of one event with threshold/AIC picks and theoretical mode arrivals."""
    from .pickers import AICPicker

    sim = setup.simulator
    thr_uv = float(dbae_to_volts(setup.detector.threshold_db)) * 1e6
    t_s0 = sim.arrival_times([trial.x_true], mode="S0")
    t_a0 = sim.arrival_times([trial.x_true], mode="A0")
    aic = AICPicker()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for ch, ax in enumerate(axes):
        h = trial.event.hit_on(ch)
        t = (h.waveform_start + np.arange(h.waveform.size)) / h.fs + trial.t_offset
        ax.plot(t * 1e6, h.waveform * 1e6, color=INK2, lw=0.8)
        ax.axhline(thr_uv, color=MUTED, lw=1, ls=":")
        ax.axhline(-thr_uv, color=MUTED, lw=1, ls=":")
        for tt, name in ((t_s0[ch], "S0"), (t_a0[ch], "A0")):
            ax.axvline(tt * 1e6, color=MUTED, lw=1, ls="--")
            ax.annotate(name, (tt * 1e6, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.axvline((h.arrival_time + trial.t_offset) * 1e6, color=SERIES["standard"], lw=1.5, label="threshold crossing")
        ax.axvline((aic.pick(h) + trial.t_offset) * 1e6, color=SERIES["aic"], lw=1.5, label="AIC onset")
        x_s = setup.specimen.sensor_x[ch]
        ax.set_title(f"Sensor {ch + 1} at x = {x_s:.2f} m: {h.amplitude_db:.0f} dB_AE", loc="left", fontsize=10)
        ax.set_ylabel("Voltage, µV")
        lim = max(float(np.abs(h.waveform).max()) * 1e6 * 1.15, 3 * thr_uv)
        ax.set_ylim(-lim, lim)
    t_first = min(t_s0) * 1e6
    axes[-1].set_xlim(t_first - 60, max(t_a0) * 1e6 + 250)
    axes[-1].set_xlabel("Time after source, µs")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Crack-like source at x = {trial.x_true:.3f} m, {trial.amplitude_db:.0f} dB at 10 cm "
                 f"(dotted: ±{setup.detector.threshold_db:.0f} dB threshold)", fontsize=10, color=INK)
    return _save(fig, path)


def error_vs_position_figure(result, setup, path) -> Path:
    """Estimated vs true position, one panel per method."""
    sx = setup.specimen.sensor_x
    lo, hi = float(result.x_true.min()), float(result.x_true.max())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, key in zip(axes, ("standard", "aic", "ml"), strict=True):
        est = result.estimates[key]
        ok = ~np.isnan(est)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=1)
        ax.scatter(result.x_true[ok], est[ok], s=6, color=SERIES[key], alpha=0.45, lw=0, zorder=2)
        _sensor_lines(ax, sx)
        mae = 1e3 * np.mean(np.abs(est[ok] - result.x_true[ok]))
        ax.set_title(f"{LABELS[key]}\nMAE {mae:.0f} mm", fontsize=10)
        ax.set_xlabel("True position, m")
    axes[0].set_ylabel("Located position, m")
    axes[0].set(xlim=(lo, hi), ylim=(lo - 0.02, hi + 0.02))
    fig.suptitle("Crack-like test sources, 45-85 dB: located vs true position (S1, S2 = sensors)",
                 fontsize=10, color=INK, y=1.04)
    return _save(fig, path)


def _binned(x, ok, values, edges):
    idx = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        sel = idx == i
        if sel.sum() >= 5:
            out[i] = values(sel)
    return out


def probability_vs_position_figure(result, setup, path) -> Path:
    """Probability of correct location (|error| <= tol, misses count as failures) along the specimen."""
    edges = np.arange(np.floor(result.x_true.min() * 20) / 20, result.x_true.max() + 0.05, 0.05)
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for key in ("standard", "aic", "ml"):
        est = result.estimates[key]
        good = np.abs(est - result.x_true) <= result.tol  # NaN compares False -> miss
        p = _binned(result.x_true, None, lambda s, good=good: 100 * good[s].mean(), edges)
        ax.plot(centers, p, color=SERIES[key], marker="o", ms=4, label=LABELS[key])
    _sensor_lines(ax, setup.specimen.sensor_x)
    ax.set(xlabel="True position, m", ylabel=f"Located within ±{result.tol * 1e3:.0f} mm, %", ylim=(0, 100))
    ax.set_title("Probability of correct location along the specimen (misses count as failures)", pad=16)
    ax.legend(loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.32))
    return _save(fig, path)


def error_vs_amplitude_figure(result, path) -> Path:
    """Median absolute location error vs source amplitude."""
    edges = np.arange(45, 86, 5.0)
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for key in ("standard", "aic", "ml"):
        est = result.estimates[key]
        err = np.abs(est - result.x_true) * 1e3
        ok = ~np.isnan(err)
        med = _binned(result.amplitude, None,
                      lambda s, err=err, ok=ok: np.median(err[s & ok]) if (s & ok).sum() >= 5 else np.nan, edges)
        ax.plot(centers, med, color=SERIES[key], marker="o", ms=4, label=LABELS[key])
    ax.set(xlabel="Source amplitude at 10 cm, dB_AE", ylabel="Median |error| of located events, mm",
           title="Location error vs signal amplitude", ylim=(0, None), xlim=(45, 85))
    ax.legend(loc="upper right")
    return _save(fig, path)
