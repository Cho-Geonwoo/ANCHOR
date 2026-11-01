"""Local, account-free experiment records."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def provenance():
    versions = {}
    for name in (
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "chex",
        "distrax",
        "gym",
        "numpy",
        "d4rl",
        "mujoco-py",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    sources = [
        *root.glob("*.py"),
        *root.glob("requirements*.txt"),
        *root.joinpath("wsrl").rglob("*.py"),
        *root.joinpath("experiments").rglob("*.py"),
    ]
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        git_commit=git.stdout.strip() or None,
        git_dirty=bool(dirty.stdout.strip()),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(sources)
        },
    )


class CSVLog:
    def __init__(self, path, columns):
        self.file = Path(path).open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()
