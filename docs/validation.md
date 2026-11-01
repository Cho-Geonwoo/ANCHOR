# Validation performed

Date: 2026-11-01. The original research checkout was preserved. The release was
prepared in the `codex/anchor-open-source` worktree from source commit `230a88a`.

## Checks that passed

- Resolved all 42 method/task agent configurations and validated all 210 generated
  training commands against the real CLI parser.
- Installed the pinned core stack in an isolated temporary Python 3.10 environment:
  JAX/JAXlib 0.4.25, Flax 0.8.2, Optax 0.1.9, Chex 0.1.86, Distrax 0.1.5,
  TensorFlow Probability 0.23.0, NumPy 1.26.4, and Gym 0.23.1.
- **14 unit tests passed** on CPU. Coverage includes manuscript hyperparameters,
  linear KL scheduling, KL direction and zero-weight equivalence, stopped reference
  gradients, frozen offline references, scalar coefficients with high UTD, genuine
  PORL critic reinitialization, actor/optimizer freezing with nonzero momentum,
  Monte Carlo returns and completed-episode sampling, time-limit bootstrapping,
  equal-task aggregation/SE, rejection of synthetic benchmark inputs, and missing
  evaluation detection.
- **All seven methods passed end-to-end synthetic smoke tests**, including offline
  updates (except SAC), online transitions, warmup, updates, evaluation, local logging,
  final model saving, offline-checkpoint checksum validation/reloading, and figure
  generation with reusable CSV data. The tests use small networks, two offline
  updates, and twelve online interactions; they are not benchmark performance runs.
- Redrew an exported result using only its CSV/provenance bundle, without reading
  original run folders or checkpoints.
- Resolved the documented Linux x86_64 dependency closure. Fixed source commits for
  D4RL, mj_envs, and mjrl were verified against their public repositories. The Adroit
  public link resolves to `offpolicy_hand_data.zip`; the legacy Python 3.10 CUDA JAX
  wheel and Kitchen dataset URL returned HTTP 200.
- Python syntax, CLI help, shell syntax, formatting, and whitespace checks passed.

## Scope and remaining verification

The target Linux/NVIDIA simulator stack and full 210-run benchmark have not been
executed as part of this cleanup. A native simulator startup check on this Mac could
not complete with its existing legacy MuJoCo framework/GL installation; no simulator
success is claimed from that attempt. Use `python scripts/check_env.py --download-datasets`
on the intended Linux machine before starting full runs.

Original raw learning curves and original pretrained checkpoints were not supplied.
The release follows the final manuscript's ANCHOR settings; discrepancies from old
scripts and ambiguous PORL details are documented in [reproduction.md](reproduction.md).
Consequently, passing code tests does not establish numerical reproduction of Figure 2.
The paper endpoint CSV is a transcription of reported aggregate measurements, not a
replacement for seed-level time-series evidence.

No GPU jobs, cloud training, or paid APIs were launched during cleanup. The
source-license issue identified in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
remains a separate publication check.
