# ANCHOR: KL-Anchored Adaptation for Stable Offline-to-Online RL

## Installation

The benchmark setup targets **Linux x86_64, Python 3.10, and an NVIDIA GPU**.

From the repository root:

```bash
git clone https://github.com/Cho-Geonwoo/ANCHOR.git
cd ANCHOR
conda env create -f environment.yml
conda activate anchor
python -m pip install -r requirements-lock-linux.txt
python -m pip install -e .

# CUDA 12 JAX build; skip this line for CPU-only code tests.
python -m pip install "jax[cuda12_pip]==0.4.25" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -c "import jax; print(jax.__version__, jax.devices())"
```

### MuJoCo and environments

On Ubuntu, install the system libraries needed by `mujoco-py`:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf
mkdir -p "$HOME/.mujoco"
curl -L https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
  -o /tmp/mujoco210-linux-x86_64.tar.gz
tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export MUJOCO_GL=egl
bash scripts/install_benchmarks.sh
```


### Datasets

D4RL AntMaze and Kitchen datasets are downloaded automatically on their first use.
Adroit uses the same public demonstration archive as WSRL and Cal-QL:

```bash
python scripts/download_adroit.py --data-dir data/adroit
export DATA_DIR_PREFIX="$(pwd)/data/adroit"

# Reset and step all six tasks, verify Adroit files, and fetch the D4RL datasets.
python scripts/check_env.py --download-datasets
```

The four Adroit files must be under `$DATA_DIR_PREFIX/offpolicy_hand_data/`:
`pen2_sparse.npy`, `pen_bc_sparse4.npy`, `door2_sparse.npy`, and `door_bc_sparse4.npy`.
The downloader records file hashes and the source URL. If automated downloading is
unavailable, obtain the [public archive](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view)
and place those four files there.

## Run ANCHOR

Train one complete offline-to-online run on Kitchen:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --method anchor --env kitchen-partial-v0 --seed 10
```

The other Figure 2 tasks use their own published settings automatically:

```bash
python train.py --method anchor --env kitchen-mixed-v0 --seed 10
python train.py --method anchor --env door-binary-v0 --seed 10
python train.py --method anchor --env pen-binary-v0 --seed 10
python train.py --method anchor --env antmaze-large-play-v2 --seed 10
python train.py --method anchor --env antmaze-large-diverse-v2 --seed 10
```

## Run the baselines

The same entry point selects the baseline and its task-specific defaults:

```bash
python train.py --method wsrl  --env kitchen-partial-v0 --seed 10
python train.py --method porl  --env kitchen-partial-v0 --seed 10
python train.py --method calql --env kitchen-partial-v0 --seed 10
python train.py --method cql   --env kitchen-partial-v0 --seed 10
python train.py --method iql   --env kitchen-partial-v0 --seed 10
python train.py --method sac   --env kitchen-partial-v0 --seed 10
```

## Reproduce Figure 2

The full grid contains **7 methods × 6 environments × 5 seeds = 210 runs**.

```bash
# Inspect every command without importing a simulator or starting training.
python scripts/run_figure2.py --dry-run

# Train all methods and tasks with seeds 10, 20, 30, 40, 50.
CUDA_VISIBLE_DEVICES=0 python scripts/run_figure2.py --skip-completed

# Plot all six task curves and their equally weighted aggregate.
python plot_figure2.py --log-dir runs --output-dir results/figure2
```

To run just ANCHOR across all six tasks and all five seeds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_figure2.py --methods anchor --skip-completed
python plot_figure2.py --methods anchor --log-dir runs --output-dir results/anchor
```

To reproduce a single benchmark with all seven methods:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_figure2.py \
  --envs kitchen-partial-v0 --skip-completed
python plot_figure2.py --envs kitchen-partial-v0 \
  --log-dir runs --output-dir results/kitchen-partial
```


## Logs and checkpoints

A default run writes to `runs/<method>/<environment>/seed<seed>/`:

| File | Contents |
| --- | --- |
| `run.json` | Resolved arguments, agent configuration, package versions, device, seed, run status |
| `episodes.csv` | Individual evaluation episodes, raw return, length, and success fraction |
| `evaluations.csv` | Per-run mean of each evaluation, including online step 0 |
| `training.jsonl` | Optimization diagnostics at each logging interval |
| `pretrained.msgpack` + `.json` | Offline checkpoint, identity, and checksum |
| `final.msgpack` | Final model state |

Success rates are unitless fractions in [0, 1]. Kitchen reports the fraction of four
stages solved; Adroit reports episode goal achievement; AntMaze uses D4RL's normalized
episode return. Plot output also contains `raw_episodes.csv`, `raw_evaluations.csv`,
`seed_curves.csv`, `curves.csv`, `provenance.json`, and `DATA_README.md` for offline reuse.

To pretrain separately and later fine-tune the same method/env/seed:

```bash
python train.py --method anchor --env kitchen-partial-v0 --seed 10 \
  --pretrain-only --output-dir pretrained/anchor/kitchen-partial-v0/seed10
python train.py --method anchor --env kitchen-partial-v0 --seed 10 \
  --checkpoint pretrained/anchor/kitchen-partial-v0/seed10/pretrained.msgpack
```


## Acknowledgments and attribution

The implementation builds on [WSRL](https://github.com/zhouzypaul/wsrl), which credits
[jaxrl_minimal](https://github.com/dibyaghosh/jaxrl_minimal). PORL is our reimplementation.
Environment and dataset dependencies retain their upstream terms. Source lineage and
available license information are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{cho2026anchor,
  title={ANCHOR: KL-Anchored Adaptation for Stable Offline-to-Online RL},
  author={Cho, Geonwoo and Im, Jaegyun and Kim, Doyoon and Li, Lexin},
  year={2026},
  note={CoRL Workshop on Continually Self-Improving Robots}
}
```
