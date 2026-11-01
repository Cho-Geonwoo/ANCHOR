"""Run the Figure 2 grid sequentially on the selected device."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wsrl.benchmark import ENVS, METHODS, SEEDS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--envs", nargs="+", choices=ENVS, default=ENVS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--log-dir", type=Path, default=Path("runs"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args(argv)
    for name in ("methods", "envs", "seeds"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            p.error(f"Duplicate --{name}")
    for env in args.envs:
        for method in args.methods:
            for seed in args.seeds:
                out = args.log_dir.resolve() / method / env / f"seed{seed}"
                metadata = out / "run.json"
                if args.skip_completed and metadata.exists():
                    saved = json.loads(metadata.read_text())
                    if saved.get("status") == "complete" and saved.get("data_kind") == "benchmark":
                        # Only skip runs made with the full published protocol.
                        from wsrl.benchmark import protocol

                        spec = protocol(method, env)
                        actual = saved.get("arguments", {})
                        keys = (
                            "offline_steps",
                            "online_steps",
                            "warmup_steps",
                            "batch_size",
                            "utd",
                            "eval_interval",
                            "eval_episodes",
                        )
                        if (saved.get("method"), saved.get("env"), saved.get("seed")) == (
                            method,
                            env,
                            seed,
                        ) and all(actual.get(k) == spec[k] for k in keys):
                            print(f"Completed: {out}", flush=True)
                            continue
                cmd = [
                    sys.executable,
                    str(ROOT / "train.py"),
                    "--method",
                    method,
                    "--env",
                    env,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(out),
                ]
                print(shlex.join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
