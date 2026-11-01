"""Exercise every method on tiny synthetic dynamics, including checkpoint reuse."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    cases = [
        ("anchor", "kitchen-partial-v0"),
        ("wsrl", "antmaze-large-play-v2"),
        ("porl", "door-binary-v0"),
        ("calql", "kitchen-partial-v0"),
        ("cql", "antmaze-large-diverse-v2"),
        ("iql", "pen-binary-v0"),
        ("sac", "kitchen-mixed-v0"),
    ]
    for method, env in cases:
        out = args.output_dir.resolve() / method / env / "seed10"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "train.py"),
                "--smoke-test",
                "--method",
                method,
                "--env",
                env,
                "--output-dir",
                str(out),
            ],
            cwd=ROOT,
            check=True,
        )
    checkpoint = args.output_dir.resolve() / "anchor/kitchen-partial-v0/seed10/pretrained.msgpack"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "train.py"),
            "--smoke-test",
            "--method",
            "anchor",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(args.output_dir.resolve() / "checkpoint_reuse"),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot_figure2.py"),
            "--log-dir",
            str(args.output_dir.resolve()),
            "--output-dir",
            str(args.output_dir.resolve() / "plot"),
            "--methods",
            "anchor",
            "--envs",
            "kitchen-partial-v0",
            "--seeds",
            "10",
            "--max-step",
            "12",
            "--allow-synthetic",
        ],
        cwd=ROOT,
        check=True,
    )
    print(
        "All method/transition/checkpoint/plot smoke tests passed. These are synthetic tests only."
    )


if __name__ == "__main__":
    main()
