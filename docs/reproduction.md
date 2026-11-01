# Figure 2 reproduction protocol

The final manuscript supplied with the release preparation is the source of truth for
ANCHOR's method, task list, network sizes, hyperparameters, and budgets. Historical
launchers include multiple incompatible sweeps. The author explicitly selected the
manuscript values for this cleanup on November 1, 2026.

## ANCHOR

- Offline pretraining uses CQL, with coefficient 5 for Kitchen/AntMaze and 1 for Adroit.
- Online critic regularization is fixed: 1 for both Kitchen tasks and 0 for the other
  four tasks. It is not exponentially annealed from the offline value.
- Actor regularization is forward KL, `KL(current policy || frozen offline policy)`.
  For tanh Gaussian policies, KL is computed in the underlying Gaussian space; both
  distributions use the same bijection. Reference parameters have no gradient.
- The KL coefficient is linear, reaches exactly zero at the task's published duration,
  and remains zero. Its clock is collected online environment steps, including the
  5,000-step collection-only warmup; UTD does not accelerate this clock.
- The offline data array is released before online collection. The frozen model is
  retained; offline observations/actions are never inserted into online replay.
- ANCHOR, CQL, and Cal-QL retain the WSRL CQL backup implementation. WSRL/PORL switch
  to the unconstrained backup at the online boundary.

## Baseline conventions

| Method | Offline learner | Online behavior |
| --- | --- | --- |
| WSRL | CQL | Keep pretrained actor/critic; no conservative loss; UTD 16 |
| PORL | CQL actor | Fresh critic; pretrained actor frozen through critic warmup; UTD 16 |
| Cal-QL | Cal-QL | Calibrated conservative penalty, including online MC returns |
| CQL | CQL | Retain conservative penalty |
| IQL | IQL | Retain expectile value loss and advantage-weighted actor |
| SAC | None | Random-action collection warmup, then online learning |

The inherited WSRL AntMaze baseline configs autotune the CQL coefficient against a
0.8 action gap. ANCHOR uses the fixed offline coefficient in the manuscript table.
Cal-QL and CQL retain their baseline coefficient settings online. The IQL actor uses
the WSRL learning rate 3e-4; the 1e-4 actor rate in the paper's ANCHOR table applies to
ANCHOR and the SAC/CQL-based agents. All critics use 3e-4.

Batch size 512 follows the inherited update API: for UTD `u`, the full batch is split
into `u` critic minibatches, followed by one actor/temperature update on the full batch.
Thus UTD 16 uses critic minibatches of 32, with 16 critic updates per interaction.
There is no hidden multiplication of the configured batch size by UTD.

The cleanup also fixes an inherited optimizer issue: replacing an excluded network's
loss with zero still applied its old Adam momentum. Critic-only passes now preserve
the actor's parameters and optimizer state exactly, even after offline pretraining.
This is tested with nonzero actor momentum. It restores the stated update schedule,
and is another reason this release is not a byte-for-byte replay of historical runs.

The manuscript has two unresolved PORL protocol statements: its baseline description
calls for critic training during initial sampling, while its hyperparameter paragraph
says all main methods use collection-only warmup. This release retains PORL's defining
critic-only warmup, uses the specified **5,000** interactions, and records that choice
here. Its old launchers used 20,000 for some domains. The historical code's actor-only
restore also discarded the returned reinitialized agent; the release explicitly
constructs a fresh critic and tests that the transferred actor is unchanged. Warmup
exploration uses the inherited stochastic pretrained policy; the paper's PORL description
mentions epsilon-greedy sampling but does not specify an epsilon. No extra epsilon
parameter has been invented.

## Differences from old ANCHOR sweeps

Some historical scripts used Kitchen online alpha 0 with exponential alpha decay,
a final KL coefficient of 0.5, door beta0 0.5, or diverse-maze beta0 0.2. The final
manuscript instead specifies Kitchen online alpha 1 with a fixed coefficient, final
KL coefficient 0, door beta0 0.2, and diverse-maze beta0 0.4. The release follows those
manuscript settings, as requested. Its named ANCHOR implementation uses the stated
CQL backbone; old scripts frequently loaded Cal-QL checkpoints instead.

The original main-result table also reports different initial performance across
methods. This release trains each requested method/seed from scratch and does not
claim that a generated checkpoint is identical to any unpublished original checkpoint.

## Measurement and plotting

- Seeds: 10, 20, 30, 40, 50. Separate training and evaluation environment seeds; the
  evaluation environment uses seed + 1000. Sampling uses JAX/NumPy seeded generators.
- Evaluation: 20 deterministic episodes at the online boundary and every 5,000 online
  environment steps through 400,000, including warmup in the interaction budget.
- Kitchen: final cumulative stage count / 4. Adroit: whether the goal was achieved at
  least once in an episode. AntMaze: D4RL normalized episode return.
- Each task's displayed mean and SE are computed across run/seed evaluation means.
  The aggregate first takes the arithmetic mean of tasks within each matching seed,
  then computes the mean and sample SE across seeds.
- Bands are one SE, clipped to [0,1]. A single seed has no SE band. No smoothing,
  interpolation, resampling, bootstrap, or performance-based exclusion is used.
- Missing runs, incomplete runs, missing evaluations, duplicate evaluation episodes,
  and disagreements between episode data and logged evaluation means are errors.

`raw_episodes.csv` preserves all available episode observations. `seed_curves.csv`
preserves per-run observations and task averages. `curves.csv` contains the exact
plotted values. `provenance.json` records configurations, input hashes, and processing.
These outputs can be reused offline without retraining or account credentials.

## Verification scope

The code tests validate the implementation and workflow, not convergence to the
published results. See [validation.md](validation.md) for the tests actually run.
The manuscript endpoint CSV contains reported aggregates only. Original seed-level
curve data were absent, so the exact published Figure 2 curves cannot be reconstructed
offline from that endpoint CSV.
