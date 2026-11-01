"""Redraw an exported figure bundle offline, without original run folders or models."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    from wsrl.plotting import draw

    metadata = json.loads((args.data_dir / "provenance.json").read_text())
    with (args.data_dir / "curves.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["online_step"] = int(row["online_step"])
        row["n_seeds"] = int(row["n_seeds"])
        for key in ("mean", "standard_error", "band_lower", "band_upper"):
            row[key] = float(row[key]) if row[key] else None
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "raw_evaluations.csv",
        "raw_episodes.csv",
        "seed_curves.csv",
        "curves.csv",
        "provenance.json",
        "DATA_README.md",
    ):
        shutil.copy2(args.data_dir / name, out / name)
    draw(
        rows,
        metadata["methods"],
        metadata["envs"],
        out,
        synthetic=metadata["data_kind"] != "benchmark",
    )
    print(f"Redrew the exported CSV data in {out.resolve()}")


if __name__ == "__main__":
    main()
