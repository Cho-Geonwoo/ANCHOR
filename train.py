"""Train one Figure 2 method from scratch, without private checkpoints or services."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from wsrl.benchmark import ENVS, METHODS, TASKS, agent_config, kl_weight, protocol


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="anchor")
    p.add_argument("--env", choices=ENVS, default="kitchen-partial-v0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Offline pretrained.msgpack to transfer")
    p.add_argument("--pretrain-only", action="store_true")
    p.add_argument("--offline-steps", type=int)
    p.add_argument("--online-steps", type=int, default=400_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--utd", type=int)
    p.add_argument("--eval-interval", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=5_000)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved protocol without importing JAX/MuJoCo",
    )
    p.add_argument(
        "--smoke-test", action="store_true", help="Tiny synthetic integration test; not a benchmark"
    )
    args = p.parse_args(argv)
    spec = protocol(args.method, args.env)
    if args.smoke_test:
        args.offline_steps = 0 if args.method == "sac" else 2
        args.online_steps, args.warmup_steps, args.batch_size = 12, 4, 8
        args.eval_interval, args.eval_episodes, args.log_interval = 4, 2, 4
        args.utd = 1 if args.method == "iql" else 2
    if args.offline_steps is None:
        args.offline_steps = spec["offline_steps"]
    if args.utd is None:
        args.utd = spec["utd"]
    for key in ("batch_size", "utd", "eval_interval", "eval_episodes", "log_interval"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if min(args.offline_steps, args.online_steps, args.warmup_steps) < 0:
        p.error("Training step counts must be nonnegative")
    if args.batch_size % args.utd:
        p.error("--batch-size must be divisible by --utd")
    if args.method == "iql" and args.utd != 1:
        p.error("Figure 2 IQL uses --utd 1")
    if args.method == "sac" and (args.offline_steps or args.checkpoint or args.pretrain_only):
        p.error("SAC starts online from scratch; no offline phase or checkpoint")
    if args.method != "sac" and args.offline_steps == 0:
        p.error("Offline-to-online methods require positive --offline-steps")
    if args.output_dir is None:
        prefix = Path("runs/_smoke" if args.smoke_test else "runs")
        args.output_dir = prefix / args.method / args.env / f"seed{args.seed}"
    return args


def make_online_agent(agent, method, config, observations, actions, seed):
    """Keep ANCHOR's reference immutable and make PORL's critic genuinely fresh."""
    import jax
    from flax.core import FrozenDict, freeze, unfreeze
    from wsrl.agents import agents

    if method == "porl":
        fresh = agents[method].create(
            jax.random.PRNGKey(seed + 100_003), observations, actions, encoder_def=None, **config
        )
        params, targets = unfreeze(fresh.state.params), unfreeze(fresh.state.target_params)
        params["modules_actor"] = agent.state.params["modules_actor"]
        targets["modules_actor"] = agent.state.params["modules_actor"]
        # Flax versions differ in whether Module.init returns dict or FrozenDict.
        # Preserve the type used to initialize optimizer moments.
        if isinstance(fresh.state.params, FrozenDict):
            params, targets = freeze(params), freeze(targets)
        agent = fresh.replace(state=fresh.state.replace(params=params, target_params=targets))
    if method in ("anchor", "wsrl", "porl", "cql", "calql"):
        online = dict(agent.config)
        if method in ("wsrl", "porl"):
            online.update(use_cql_loss=False, wsrl_online_mode=True)
        else:
            alpha = config["online_cql_alpha"]
            online.update(
                cql_alpha=alpha,
                cql_autotune_alpha=False if method == "anchor" else online["cql_autotune_alpha"],
                use_cql_loss=alpha > 0,
            )
        agent = agent.replace(config=freeze(online))
    return agent


def run(args):
    import hashlib
    import math
    import time
    from dataclasses import asdict
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from wsrl.agents import agents
    from wsrl.agents.anchor import gaussian_parameters
    from wsrl.data.dataset import Dataset
    from wsrl.data.replay_buffer import ReplayBuffer, ReplayBufferMC
    from wsrl.logging import CSVLog, provenance, write_json

    np.random.seed(args.seed)
    task = TASKS[args.env]
    config = agent_config(args.method, args.env, small=args.smoke_test)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)

    # JSON represents unbounded clipping limits as strings, not nonstandard Infinity literals.
    def config_json(value):
        if isinstance(value, dict):
            return {k: config_json(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [config_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    identity = dict(
        method=args.method,
        env=args.env,
        seed=args.seed,
        offline_steps=args.offline_steps,
        agent_config=config_json(config),
        data_kind="synthetic_smoke_test" if args.smoke_test else "benchmark",
    )
    record = dict(
        **identity,
        task=asdict(task),
        status="running",
        provenance=provenance(),
        devices=[str(d) for d in jax.devices()],
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    write_json(out / "run.json", record)
    episodes = CSVLog(
        out / "episodes.csv",
        ["phase", "offline_step", "online_step", "episode", "return", "length", "success_rate"],
    )
    metrics = CSVLog(
        out / "evaluations.csv",
        ["phase", "offline_step", "online_step", "success_rate", "return", "num_episodes"],
    )
    train_log = (out / "training.jsonl").open("w")
    env = eval_env = None
    started = time.monotonic()
    try:
        if args.smoke_test:
            from wsrl.smoke import SmokeEnv, dataset

            env, eval_env = SmokeEnv(args.seed), SmokeEnv(args.seed + 1000)
            offline = None if args.checkpoint else Dataset(dataset(args.seed), args.seed)
        else:
            from wsrl.envs.env_common import make_gym_env

            env = make_gym_env(
                args.env,
                reward_scale=task.reward_scale,
                reward_bias=task.reward_bias,
                seed=args.seed,
            )
            eval_env = make_gym_env(args.env, seed=args.seed + 1000)
            offline = None
            if args.method != "sac" and not args.checkpoint:
                if task.domain == "adroit":
                    from wsrl.envs.adroit_binary_dataset import get_hand_dataset_with_mc_calculation

                    data = get_hand_dataset_with_mc_calculation(
                        args.env,
                        config["discount"],
                        reward_scale=task.reward_scale,
                        reward_bias=task.reward_bias,
                        clip_action=0.99999,
                    )
                elif args.method == "calql":
                    from wsrl.envs.d4rl_dataset import get_d4rl_dataset_with_mc_calculation

                    data = get_d4rl_dataset_with_mc_calculation(
                        args.env, task.reward_scale, task.reward_bias, 0.99999, config["discount"]
                    )
                else:
                    from wsrl.envs.d4rl_dataset import get_d4rl_dataset

                    data = get_d4rl_dataset(
                        args.env, task.reward_scale, task.reward_bias, clip_action=0.99999
                    )
                offline = Dataset(data, args.seed)
                del data
        obs_example = jnp.zeros((1, *env.observation_space.shape))
        act_example = jnp.zeros((1, *env.action_space.shape))
        agent = agents[args.method].create(
            jax.random.PRNGKey(args.seed), obs_example, act_example, encoder_def=None, **config
        )

        def evaluate(phase, offline_step, online_step):
            scores, returns = [], []
            for episode in range(args.eval_episodes):
                obs, _ = eval_env.reset()
                rewards, infos = [], []
                while True:
                    action = np.asarray(agent.sample_actions(obs, argmax=True))
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    rewards.append(float(reward))
                    infos.append(info)
                    if terminated or truncated:
                        break
                if args.smoke_test or task.domain == "adroit":
                    score = float(any(i.get("goal_achieved", False) for i in infos))
                elif task.domain == "kitchen":
                    score = rewards[-1] / 4.0
                else:
                    score = float(eval_env.get_normalized_score(sum(rewards)))
                if not math.isfinite(score) or not 0 <= score <= 1.00001:
                    raise ValueError(f"Invalid success rate {score}; check environment version")
                scores.append(score)
                returns.append(sum(rewards))
                episodes.write(
                    {
                        "phase": phase,
                        "offline_step": offline_step,
                        "online_step": online_step,
                        "episode": episode,
                        "return": sum(rewards),
                        "length": len(rewards),
                        "success_rate": score,
                    }
                )
            row = {
                "phase": phase,
                "offline_step": offline_step,
                "online_step": online_step,
                "success_rate": float(np.mean(scores)),
                "return": float(np.mean(returns)),
                "num_episodes": args.eval_episodes,
            }
            metrics.write(row)
            display_step = online_step if phase == "online" else offline_step
            print(
                f"{args.method} {args.env} seed={args.seed} {phase} step={display_step} "
                f"success={row['success_rate']:.3f}",
                flush=True,
            )

        def log_update(info, phase, step):
            flat = {}

            def collect(values, prefix=""):
                for key, value in values.items():
                    name = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        collect(value, name)
                    elif value is not None:
                        mean = float(np.asarray(jax.device_get(value)).mean())
                        if not math.isfinite(mean):
                            raise FloatingPointError(f"Nonfinite {name} at {phase} step {step}")
                        flat[name] = mean

            collect(info)
            train_log.write(
                json.dumps(dict(phase=phase, step=step, metrics=flat), allow_nan=False) + "\n"
            )
            train_log.flush()

        if args.checkpoint:
            checkpoint = args.checkpoint.resolve()
            saved = json.loads(checkpoint.with_suffix(".json").read_text())
            if saved["identity"] != identity:
                raise ValueError(
                    "Checkpoint protocol differs from this run (method/env/seed/config/steps)"
                )
            payload = checkpoint.read_bytes()
            if hashlib.sha256(payload).hexdigest() != saved["sha256"]:
                raise ValueError("Checkpoint checksum mismatch")
            agent = serialization.from_bytes(agent, payload)
        else:
            for step in range(1, args.offline_steps + 1):
                agent, info = agent.update(offline.sample(args.batch_size))
                if step % args.log_interval == 0 or step == args.offline_steps:
                    log_update(info, "offline", step)
                if step % args.eval_interval == 0 and step != args.offline_steps:
                    evaluate("offline", step, 0)
            if args.offline_steps:
                payload = serialization.to_bytes(agent)
                temp = out / "pretrained.msgpack.tmp"
                temp.write_bytes(payload)
                os.replace(temp, out / "pretrained.msgpack")
                write_json(
                    out / "pretrained.json",
                    dict(
                        identity=identity,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        offline_steps=args.offline_steps,
                    ),
                )
        if args.pretrain_only:
            evaluate("offline", args.offline_steps, 0)
            record["status"] = "pretrained"
            return

        reference = agent if args.method == "anchor" else None
        agent = make_online_agent(agent, args.method, config, obs_example, act_example, args.seed)
        evaluate("online", args.offline_steps, 0)
        # Offline observations and transitions are released before online interaction.
        del offline
        buffer_cls = ReplayBufferMC if args.method == "calql" else ReplayBuffer
        buffer_args = dict(
            observation_space=env.observation_space,
            action_space=env.action_space,
            capacity=args.online_steps + 1,
            seed=args.seed,
            discount=config["discount"],
        )
        if args.method == "calql":
            buffer_args.update(
                env_name=args.env, reward_scale=task.reward_scale, reward_bias=task.reward_bias
            )
        replay = buffer_cls(**buffer_args)
        observation, _ = env.reset()
        rng = jax.random.PRNGKey(args.seed + 2000)
        for step in range(1, args.online_steps + 1):
            rng, action_rng = jax.random.split(rng)
            if args.method == "sac" and step <= args.warmup_steps:
                action = np.asarray(
                    jax.random.uniform(action_rng, env.action_space.shape, minval=-1.0, maxval=1.0)
                )
            else:
                action = np.asarray(agent.sample_actions(observation, seed=action_rng))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay.insert(
                dict(
                    observations=observation,
                    next_observations=next_obs,
                    actions=action,
                    rewards=reward,
                    masks=1.0 - terminated,
                    dones=float(terminated or truncated),
                )
            )
            observation = next_obs
            if terminated or truncated:
                observation, _ = env.reset()
            ready = (
                bool(replay._allow_idxs)
                if args.method == "calql"
                else len(replay) >= min(args.batch_size, args.warmup_steps)
            )
            info = None
            if ready and (step > args.warmup_steps or args.method == "porl"):
                batch = replay.sample(args.batch_size)
                if args.method == "anchor":
                    dist = reference.forward_policy(batch["observations"], train=False)
                    loc, scale = gaussian_parameters(dist)
                    batch.update(
                        reference_policy_loc=loc,
                        reference_policy_stds=scale,
                        kl_reg_weight=jnp.asarray(
                            kl_weight(task.beta0, step, task.kl_steps), jnp.float32
                        ),
                    )
                if args.method == "porl" and step <= args.warmup_steps:
                    agent, info = agent.update_high_utd_critic(batch, utd_ratio=args.utd)
                elif args.utd > 1:
                    agent, info = agent.update_high_utd(batch, utd_ratio=args.utd)
                else:
                    agent, info = agent.update(batch)
            if info is not None and (step % args.log_interval == 0 or step == args.online_steps):
                log_update(info, "online", step)
            if step % args.eval_interval == 0 or step == args.online_steps:
                evaluate("online", args.offline_steps, step)
        (out / "final.msgpack").write_bytes(serialization.to_bytes(agent))
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "run.json", record)
        episodes.close()
        metrics.close()
        train_log.close()
        for environment in (env, eval_env):
            if environment is not None:
                environment.close()


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        resolved = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        print(json.dumps(dict(**resolved, task=vars(TASKS[args.env])), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
