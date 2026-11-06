"""Command-line interface: ``aelocate {gui,benchmark,stream,train}``."""

from __future__ import annotations

import argparse
import sys
import time


def _cmd_gui(args) -> int:
    from .gui.app import main as gui_main

    return gui_main(preset=args.preset, model=args.model)


def _cmd_screenshot(args) -> int:
    from .gui.app import render_screenshot

    print("wrote", render_screenshot(args.preset, args.out, seconds=args.seconds, train=not args.no_ml))
    return 0


def _cmd_benchmark(args) -> int:
    from .benchmark import run_benchmark, write_report
    from .presets import PRESETS

    setup = PRESETS[args.preset]()
    result = run_benchmark(setup, n_train=args.n_train, n_test=args.n_test, tol=args.tol / 1e3, seed=args.seed)
    print(result.table("all"))
    if args.out:
        for p in write_report(result, setup, args.out):
            print("wrote", p)
    return 0


def _cmd_train(args) -> int:
    from .benchmark import train_ml_locator
    from .presets import PRESETS

    setup = PRESETS[args.preset]()
    ml, velocities = train_ml_locator(setup, n=args.n, seed=args.seed)
    ml.save(args.out)
    print(f"velocity {velocities['standard']:.0f} m/s; model saved to {args.out}")
    return 0


def _cmd_stream(args) -> int:
    """Headless real-time run: prints located events as JSON lines (pipe into anything)."""
    import json

    from .daq import AcquisitionSimulator
    from .pipeline import ProcessingChain
    from .presets import PRESETS
    from .scenarios import default_emitters

    setup = PRESETS[args.preset]()
    daq = AcquisitionSimulator(setup.simulator, default_emitters(setup), block_size=int(setup.fs * 0.02),
                               seed=args.seed)
    v = setup.group_velocity("S0")
    chain = ProcessingChain.from_setup(setup, setup.locator(v))
    t_wall = time.perf_counter()
    n_samples = 0
    while daq.time < args.seconds:
        block = daq.next_block()
        n_samples += block.data.size
        for loc in chain.process(block).located:
            print(json.dumps({"t": round(loc.time, 6), "x": round(loc.x, 4),
                              "y": None if loc.y is None else round(loc.y, 4),
                              "amp_db": round(loc.event.first.amplitude_db, 1), "channels": loc.channels}))
    dt = time.perf_counter() - t_wall
    print(f"# processed {args.seconds:.1f} s of {setup.n_channels}-channel data in {dt:.2f} s "
          f"({n_samples / dt / 1e6:.1f} MS/s)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    from .presets import PRESETS

    p = argparse.ArgumentParser(prog="aelocate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gui", help="real-time acquisition and location monitor (PySide6)")
    g.add_argument("--preset", choices=sorted(PRESETS), default="plate-linear")
    g.add_argument("--model", help="trained ML locator (.pkl) from 'aelocate train'")
    g.set_defaults(func=_cmd_gui)

    sc = sub.add_parser("screenshot", help="run the monitor headless and save a PNG of the window")
    sc.add_argument("--preset", choices=sorted(PRESETS), default="plate-linear")
    sc.add_argument("--seconds", type=float, default=20.0, help="stream time to simulate before the capture")
    sc.add_argument("--no-ml", action="store_true", help="skip training the neural network")
    sc.add_argument("--out", default="aelocate-gui.png")
    sc.set_defaults(func=_cmd_screenshot)

    b = sub.add_parser("benchmark", help="location accuracy: standard vs AIC vs neural network")
    b.add_argument("--preset", choices=sorted(PRESETS), default="plate-linear")
    b.add_argument("--n-train", type=int, default=6000)
    b.add_argument("--n-test", type=int, default=3000)
    b.add_argument("--tol", type=float, default=25.0, help="correct-location tolerance, mm")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out", help="directory for figures and results.md")
    b.set_defaults(func=_cmd_benchmark)

    t = sub.add_parser("train", help="train and save the ML locator for a preset")
    t.add_argument("--preset", choices=sorted(PRESETS), default="plate-linear")
    t.add_argument("--n", type=int, default=4000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="ml_locator.pkl")
    t.set_defaults(func=_cmd_train)

    s = sub.add_parser("stream", help="headless streaming run, located events as JSON lines")
    s.add_argument("--preset", choices=sorted(PRESETS), default="plate-linear")
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=_cmd_stream)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
