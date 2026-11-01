# ANCHOR: KL-Anchored Adaptation for Stable Offline-to-Online RL

**CoRL 2026 Workshop on Continually Self-Improving Robots**
Geonwoo Cho, Jaegyun Im, Doyoon Kim, Lexin Li

ANCHOR stabilizes offline-to-online reinforcement learning by anchoring the actor to the
frozen offline policy, linearly annealing the KL coefficient, and keeping a small,
fixed conservative critic penalty. Online training uses newly collected experience only.

This repository contains the six benchmarks and seven methods in **Figure 2**:
**ANCHOR, WSRL, PORL, Cal-QL, CQL, IQL, and SAC**. A run starts from scratch, performs
its offline pretraining, and then collects 400,000 online environment steps. No private
checkpoint, W&B account, or API credential is needed.

## Installation

The benchmark setup targets **Linux x86_64, Python 3.10, and an NVIDIA GPU**. The
six tasks are simulated control benchmarks using the legacy D4RL/Adroit environments
from the paper. Use these versions
rather than substituting Gymnasium, Minari, or a newer task revision.

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

The core dependencies are pinned in [requirements.txt](requirements.txt). The compatible
Flax/Optax versions fix an import error in the old research requirements with JAX 0.4.25.
The Linux dependency closure is frozen in [requirements-lock-linux.txt](requirements-lock-linux.txt);
its input is [requirements-benchmarks.txt](requirements-benchmarks.txt). Simulator source
repositories are pinned to exact commits separately.

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

The installer fetches the WSRL D4RL fork (including its Kitchen reward fix),
`mjrl`, and the Adroit binary-task fork of `mj_envs`, using public HTTPS URLs.
It checks out fixed commits and their recorded submodules; it does not update them to
arbitrary newer versions. Re-export the environment variables in new terminals.

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

| Environment | Offline updates | Online critic weight | Initial KL weight | KL duration | UTD |
| --- | ---: | ---: | ---: | ---: | ---: |
| kitchen-partial-v0 | 250,000 | 1.0 | 1.0 | 160,000 | 1 |
| kitchen-mixed-v0 | 250,000 | 1.0 | 1.0 | 160,000 | 1 |
| door-binary-v0 | 40,000 | 0.0 | 0.2 | 40,000 | 16 |
| pen-binary-v0 | 40,000 | 0.0 | 1.0 | 80,000 | 1 |
| antmaze-large-play-v2 | 1,000,000 | 0.0 | 0.2 | 80,000 | 1 |
| antmaze-large-diverse-v2 | 1,000,000 | 0.0 | 0.4 | 80,000 | 1 |

All use batch size 512, 10 critics with two-critic target subsampling, layer
normalization, a 5,000-step warmup, and a 400,000-step online budget. For ANCHOR,
`beta(t) = beta0 * max(1 - t / T_KL, 0)` counts online environment steps, including
warmup. The online critic coefficient is fixed; there is no critic-coefficient sweep.
The complete settings are in [wsrl/benchmark.py](wsrl/benchmark.py).

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

SAC starts online from scratch. WSRL switches a pretrained CQL agent to an
unconstrained online backup. PORL transfers only the pretrained actor, initializes a
fresh critic, and updates only the critic during warmup. WSRL and PORL use UTD 16;
CQL, Cal-QL, IQL, and SAC use UTD 1. All methods use **zero offline replay** online.
See [Reproduction details](docs/reproduction.md) for baseline conventions and the
remaining ambiguities in the historical experiment records.

## Reproduce Figure 2

The full grid contains **7 methods × 6 environments × 5 seeds = 210 runs**.
Commands execute sequentially, so a single process uses one selected device at a time.
The full benchmark is a substantial training workload; start with one run above.

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

Outputs include `figure2.pdf`, `figure2.png`, every evaluation episode, per-seed
curves, and the exact means/standard errors and band endpoints drawn in the figure.
No smoothing or interpolation is applied. The default plot requires every requested
run and evaluation point to be present and complete. The shaded band is one standard
error across seeds; the aggregate averages tasks within each seed before aggregating
seeds. With one seed, no uncertainty band is shown.

To redraw an exported result without the original run directories or model checkpoints:

```bash
python scripts/replot_csv.py --data-dir results/figure2 --output-dir results/redrawn
```

The commands implement the final manuscript protocol and generate curves from new
runs. Original seed-level time series and pretrained checkpoints were not in the
provided research repository. The paper's reported 400K endpoint means and SEs are
preserved in [data/paper_final_success.csv](data/paper_final_success.csv) for comparison;
they are not used as input to the plotting command. Full GPU training has not been
rerun as part of this cleanup. See [validation](docs/validation.md).

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

Existing output directories are never overwritten. `--skip-completed` skips only
successful runs with the full default protocol. Move just a failed or shortened seed
directory out of its expected path before retrying the grid; completed seeds will
still be skipped. A single run can also use a fresh `--output-dir`.

To pretrain separately and later fine-tune the same method/env/seed:

```bash
python train.py --method anchor --env kitchen-partial-v0 --seed 10 \
  --pretrain-only --output-dir pretrained/anchor/kitchen-partial-v0/seed10
python train.py --method anchor --env kitchen-partial-v0 --seed 10 \
  --checkpoint pretrained/anchor/kitchen-partial-v0/seed10/pretrained.msgpack
```

Checkpoint reuse validates the method, task, seed, offline budget, network configuration,
and file checksum. Checkpoints from a different method or legacy private run are not
silently substituted.

## Code tests

Core tests use CPU JAX and require no simulator, benchmark data, or service account:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
JAX_PLATFORMS=cpu python scripts/smoke_test.py --output-dir /tmp/anchor-smoke
```

The smoke command runs all seven methods on tiny synthetic dynamics, reloads an offline
checkpoint, and exercises plotting/data export. Its figure is explicitly labeled as
synthetic and cannot enter the benchmark plot by default.

## Code structure

```text
train.py                  # offline pretraining, online adaptation, evaluation
plot_figure2.py            # figure + reusable data export
wsrl/benchmark.py         # all six task presets and method defaults
wsrl/agents/anchor.py     # ANCHOR's KL actor objective
wsrl/agents/              # CQL, Cal-QL, IQL, and SAC implementations
wsrl/envs/                # legacy environment adapters and dataset loading
scripts/                  # installation, dataset checks, full-grid runner, smoke tests
experiments/configs/      # three underlying agent configuration templates
data/                     # reported paper endpoints and their provenance
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

Questions: [Geonwoo Cho](mailto:geonwoo@umich.edu).
