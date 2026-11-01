"""Strict local-CSV aggregation with all observations and plotted values exported."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wsrl.benchmark import LABELS, protocol


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return dict(
        n_seeds=len(values),
        mean=mean,
        standard_error=se,
        band_lower=max(0.0, mean - se) if se is not None else None,
        band_upper=min(1.0, mean + se) if se is not None else None,
    )


def load_runs(log_dir, methods, envs, seeds, max_step, allow_synthetic=False):
    raw, episodes, provenance, series = [], [], [], {}
    shared_grid = None
    kinds = set()
    for method in methods:
        for env in envs:
            for seed in seeds:
                path = Path(log_dir) / method / env / f"seed{seed}"
                meta = json.loads((path / "run.json").read_text())
                if (meta["method"], meta["env"], meta["seed"]) != (method, env, seed):
                    raise ValueError(f"Run identity mismatch: {path}")
                if meta["status"] != "complete":
                    raise ValueError(f"Run is not complete: {path}")
                kind = meta["data_kind"]
                kinds.add(kind)
                if kind != "benchmark" and not allow_synthetic:
                    raise ValueError(f"Synthetic run is not a benchmark result: {path}")
                if kind == "benchmark":
                    expected = protocol(method, env)
                    for key in (
                        "offline_steps",
                        "online_steps",
                        "warmup_steps",
                        "batch_size",
                        "utd",
                        "eval_interval",
                        "eval_episodes",
                    ):
                        if meta["arguments"].get(key) != expected[key]:
                            raise ValueError(
                                f"Run differs from the Figure 2 protocol ({key}): {path}"
                            )
                if meta["arguments"]["online_steps"] < max_step:
                    raise ValueError(f"Run ends before requested step {max_step}: {path}")
                interval = meta["arguments"]["eval_interval"]
                grid = list(range(0, max_step + 1, interval))
                if grid[-1] != max_step:
                    grid.append(max_step)
                if shared_grid is None:
                    shared_grid = grid
                if grid != shared_grid:
                    raise ValueError(
                        f"Different evaluation grids; no interpolation is performed: {path}"
                    )
                with (path / "episodes.csv").open(newline="") as file:
                    erows = [
                        r
                        for r in csv.DictReader(file)
                        if r["phase"] == "online" and int(r["online_step"]) <= max_step
                    ]
                episode_map = {}
                for row in erows:
                    step, episode = int(row["online_step"]), int(row["episode"])
                    identity = (step, episode)
                    if identity in episode_map:
                        raise ValueError(f"Duplicate evaluation episode {identity}: {path}")
                    episode_map[identity] = row
                    episodes.append(dict(method=method, env=env, seed=seed, **row))
                with (path / "evaluations.csv").open(newline="") as file:
                    rows = [
                        r
                        for r in csv.DictReader(file)
                        if r["phase"] == "online" and int(r["online_step"]) <= max_step
                    ]
                if [int(r["online_step"]) for r in rows] != grid:
                    raise ValueError(f"Missing, duplicate or off-grid evaluations: {path}")
                values = []
                for row in rows:
                    step = int(row["online_step"])
                    score = float(row["success_rate"])
                    count = int(row["num_episodes"])
                    if count != meta["arguments"]["eval_episodes"]:
                        raise ValueError(f"Evaluation episode count changed: {path}")
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid success rate: {path}")
                    selected = [float(episode_map[(step, i)]["success_rate"]) for i in range(count)]
                    if not math.isclose(np.mean(selected), score, abs_tol=1e-8):
                        raise ValueError(f"Episode values do not match evaluation mean: {path}")
                    raw.append(dict(method=method, env=env, seed=seed, **row))
                    values.append(score)
                series[method, env, seed] = values
                provenance.append(
                    dict(
                        method=method,
                        env=env,
                        seed=seed,
                        files={
                            n: hashlib.sha256((path / n).read_bytes()).hexdigest()
                            for n in ("run.json", "episodes.csv", "evaluations.csv")
                        },
                        source=str(path.resolve()),
                        run=meta,
                    )
                )
    if len(kinds) != 1:
        raise ValueError("Benchmark and synthetic runs cannot be combined")
    return series, shared_grid, raw, episodes, provenance, next(iter(kinds))


def aggregate(series, grid, methods, envs, seeds):
    seed_curves, curves = [], []
    for method in methods:
        for seed in seeds:
            task_values = [np.asarray(series[method, env, seed]) for env in envs]
            series[method, "aggregate", seed] = np.mean(task_values, axis=0)
        for env in [*envs, "aggregate"]:
            for index, step in enumerate(grid):
                values = [float(series[method, env, seed][index]) for seed in seeds]
                curves.append(dict(method=method, env=env, online_step=step, **summarize(values)))
                seed_curves.extend(
                    dict(method=method, env=env, seed=seed, online_step=step, success_rate=value)
                    for seed, value in zip(seeds, values)
                )
    return seed_curves, curves


def export_csv(path, rows):
    if not rows:
        raise ValueError(f"No data for {path}")
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(curves, methods, envs, output_dir, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    palette = {
        "anchor": "#1565c0",
        "wsrl": "#e69f00",
        "porl": "#009e73",
        "calql": "#cc79a7",
        "cql": "#d55e00",
        "iql": "#777777",
        "sac": "#56b4e9",
    }
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
    for axis, env in zip(axes.flat, [*envs, "aggregate"]):
        for method in methods:
            rows = [r for r in curves if r["method"] == method and r["env"] == env]
            x, y = [r["online_step"] for r in rows], [r["mean"] for r in rows]
            axis.plot(
                x,
                y,
                color=palette[method],
                label=LABELS[method],
                linewidth=2.4 if method == "anchor" else 1.5,
            )
            if rows[0]["standard_error"] is not None:
                axis.fill_between(
                    x,
                    [r["band_lower"] for r in rows],
                    [r["band_upper"] for r in rows],
                    color=palette[method],
                    alpha=0.13,
                )
        axis.set_title(env if env != "aggregate" else f"Mean over {len(envs)} tasks")
        axis.set(xlabel="Online environment steps", ylabel="Success rate", ylim=(-0.02, 1.02))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(envs) + 1 :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)
    if synthetic:
        fig.suptitle("SYNTHETIC INTEGRATION TEST — not benchmark performance", color="#a00000")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96 if synthetic else 1))
    for suffix in ("png", "pdf"):
        fig.savefig(Path(output_dir) / f"figure2.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
