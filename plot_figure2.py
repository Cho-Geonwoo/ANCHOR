"""Rebuild Figure 2 from local experiment logs and export the underlying data."""

import argparse
import sys
from pathlib import Path

from wsrl.benchmark import ENVS, METHODS, SEEDS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--output-dir", type=Path, default=Path("results/figure2"))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--envs", nargs="+", choices=ENVS, default=ENVS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--max-step", type=int, default=400_000)
    p.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Testing only; labels the figure as synthetic",
    )
    args = p.parse_args(argv)
    for name in ("methods", "envs", "seeds"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            p.error(f"Duplicate --{name}")
    if args.max_step <= 0:
        p.error("--max-step must be positive")
    from wsrl.logging import write_json
    from wsrl.plotting import aggregate, draw, export_csv, load_runs

    series, grid, raw, episodes, sources, kind = load_runs(
        args.log_dir, args.methods, args.envs, args.seeds, args.max_step, args.allow_synthetic
    )
    seed_curves, curves = aggregate(series, grid, args.methods, args.envs, args.seeds)
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    export_csv(out / "raw_evaluations.csv", raw)
    export_csv(out / "raw_episodes.csv", episodes)
    export_csv(out / "seed_curves.csv", seed_curves)
    export_csv(out / "curves.csv", curves)
    write_json(
        out / "provenance.json",
        dict(
            data_kind=kind,
            methods=args.methods,
            envs=args.envs,
            seeds=args.seeds,
            online_steps=grid,
            sources=sources,
            command=sys.argv,
            uncertainty="sample standard deviation across seeds (ddof=1) / sqrt(n); one SE",
            band="mean +/- one SE, clipped to [0,1]; omitted for n=1",
            aggregate="unweighted mean over tasks within each matched seed, then mean and SE over seeds",
            smoothing=None,
            interpolation=None,
            exclusions=None,
            bootstrap_seed=None,
        ),
    )
    (out / "DATA_README.md").write_text(
        "# Figure data\n\nData kind: " + kind + ".\n\n"
        "`raw_episodes.csv`: all recorded online evaluation episodes (return in environment reward units; "
        "length and online_step in environment steps). `raw_evaluations.csv`: their per-run means. "
        "`seed_curves.csv`: each seed's task and equal-task-average success rates (unitless, 0–1). "
        "`curves.csv`: exact plotted means, sample SEs and clipped band endpoints. "
        "No smoothing, interpolation, bootstrap or run exclusions. Missing runs/evaluations fail validation. "
        "Input hashes, full run configurations and processing details are in `provenance.json`. "
        "Replot offline from these CSVs using `wsrl/plotting.py`; no model or service is needed.\n"
    )
    draw(curves, args.methods, args.envs, out, synthetic=kind != "benchmark")
    print(f"Saved figure2.pdf, figure2.png and reusable CSV data to {out.resolve()}")


if __name__ == "__main__":
    main()
