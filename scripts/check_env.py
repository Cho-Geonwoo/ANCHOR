"""Check installed simulators, registered tasks and Adroit dataset paths."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsrl.benchmark import ENVS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", nargs="+", choices=ENVS, default=ENVS)
    p.add_argument(
        "--download-datasets", action="store_true", help="Also download/load D4RL datasets"
    )
    args = p.parse_args()
    import jax
    import numpy as np
    from wsrl.envs.env_common import make_gym_env

    print("JAX:", jax.__version__, "Devices:", jax.devices())
    failures = []
    for name in args.envs:
        env = None
        try:
            env = make_gym_env(name, seed=10)
            obs, _ = env.reset()
            result = env.step(np.zeros(env.action_space.shape, np.float32))
            if not np.isfinite(obs).all() or not np.isfinite(result[0]).all():
                raise ValueError("Nonfinite observation")
            if "binary" in name:
                root = Path(os.environ.get("DATA_DIR_PREFIX", "~/adroit_data")).expanduser()
                task = name.split("-")[0]
                for filename in (f"{task}2_sparse.npy", f"{task}_bc_sparse4.npy"):
                    if not (root / "offpolicy_hand_data" / filename).is_file():
                        raise FileNotFoundError(root / "offpolicy_hand_data" / filename)
            elif args.download_datasets:
                data = env.env.get_dataset()
                print(name, "dataset transitions:", len(data["rewards"]))
            print(
                "OK:",
                name,
                "observation",
                env.observation_space.shape,
                "action",
                env.action_space.shape,
                "horizon",
                env.horizon,
            )
        except Exception as exc:
            failures.append(name)
            print(f"FAILED: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            if env is not None:
                env.close()
    if failures:
        raise SystemExit(f"Environment check failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
