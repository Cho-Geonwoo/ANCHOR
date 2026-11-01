"""Figure 2 protocol, using the final manuscript rather than historical sweeps."""

from dataclasses import asdict, dataclass

METHODS = ("anchor", "wsrl", "porl", "calql", "cql", "iql", "sac")
SEEDS = (10, 20, 30, 40, 50)
ENVS = (
    "kitchen-partial-v0",
    "kitchen-mixed-v0",
    "door-binary-v0",
    "pen-binary-v0",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
)
LABELS = {
    "anchor": "ANCHOR (Ours)",
    "wsrl": "WSRL",
    "porl": "PORL",
    "calql": "Cal-QL",
    "cql": "CQL",
    "iql": "IQL",
    "sac": "SAC",
}


@dataclass(frozen=True)
class Task:
    domain: str
    offline_steps: int
    reward_scale: float
    reward_bias: float
    alpha_off: float
    alpha_on: float
    beta0: float
    kl_steps: int
    actor_dims: tuple
    critic_dims: tuple
    anchor_utd: int = 1


TASKS = {
    "kitchen-partial-v0": Task(
        "kitchen", 250_000, 1.0, -4.0, 5.0, 1.0, 1.0, 160_000, (512, 512, 512), (512, 512, 512)
    ),
    "kitchen-mixed-v0": Task(
        "kitchen", 250_000, 1.0, -4.0, 5.0, 1.0, 1.0, 160_000, (512, 512, 512), (512, 512, 512)
    ),
    "door-binary-v0": Task(
        "adroit", 40_000, 10.0, 5.0, 1.0, 0.0, 0.2, 40_000, (512, 512), (512, 512, 512), 16
    ),
    "pen-binary-v0": Task(
        "adroit", 40_000, 10.0, 5.0, 1.0, 0.0, 1.0, 80_000, (512, 512), (512, 512, 512)
    ),
    "antmaze-large-play-v2": Task(
        "antmaze", 1_000_000, 10.0, -5.0, 5.0, 0.0, 0.2, 80_000, (256, 256), (256, 256, 256, 256)
    ),
    "antmaze-large-diverse-v2": Task(
        "antmaze", 1_000_000, 10.0, -5.0, 5.0, 0.0, 0.4, 80_000, (256, 256), (256, 256, 256, 256)
    ),
}


def protocol(method, env):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    task = TASKS[env]
    return dict(
        method=method,
        env=env,
        task=asdict(task),
        offline_steps=0 if method == "sac" else task.offline_steps,
        online_steps=400_000,
        warmup_steps=5_000,
        batch_size=512,
        utd=task.anchor_utd if method == "anchor" else (16 if method in ("wsrl", "porl") else 1),
        eval_interval=5_000,
        eval_episodes=20,
    )


def kl_weight(beta0, online_step, kl_steps):
    """Manuscript schedule, indexed by collected online transitions."""
    if kl_steps <= 0:
        raise ValueError("kl_steps must be positive")
    return float(beta0 * max(1.0 - max(online_step, 0) / kl_steps, 0.0))


def agent_config(method, env, small=False):
    from experiments.configs import cql_config, iql_config, sac_config

    task = TASKS[env]
    if method == "iql":
        cfg = iql_config.get_config()
        cfg.expectile = 0.9 if task.domain == "antmaze" else 0.7
        cfg.temperature = 10.0 if task.domain == "antmaze" else 0.5
        if task.domain != "antmaze":
            cfg.policy_network_kwargs.dropout_rate = 0.1
    elif method == "sac":
        cfg = sac_config.get_config()
    else:
        cfg = cql_config.get_config()
        cfg.cql_alpha = task.alpha_off
        cfg.online_cql_alpha = task.alpha_on if method == "anchor" else task.alpha_off
        cfg.cql_autotune_alpha = task.domain == "antmaze" and method != "anchor"
        cfg.cql_target_action_gap = 0.8 if task.domain == "antmaze" else 1.0
        cfg.cql_importance_sample = task.domain != "kitchen"
        cfg.use_calql = method == "calql"
    cfg.critic_ensemble_size = 10
    cfg.critic_subsample_size = 2
    cfg.policy_network_kwargs.hidden_dims = task.actor_dims
    cfg.critic_network_kwargs.hidden_dims = task.critic_dims
    for key in ("policy_network_kwargs", "critic_network_kwargs"):
        cfg[key].activations = "relu"
        cfg[key].use_layer_norm = True
        if task.domain != "kitchen":
            cfg[key].kernel_scale_final = 1e-2
    if method != "iql":
        cfg.policy_kwargs.std_parameterization = "uniform" if task.domain == "antmaze" else "exp"
        cfg.policy_kwargs.tanh_squash_distribution = True
    if small:
        cfg.policy_network_kwargs.hidden_dims = (16, 16)
        cfg.critic_network_kwargs.hidden_dims = (16, 16)
        cfg.critic_ensemble_size = 2
        if "cql_n_actions" in cfg:
            cfg.cql_n_actions = 2
    return cfg.to_dict()
