import numpy as np


class BenchmarkEnv:
    """Seeded old/new Gym adapter with explicit, reproducible episode limits.

    Legacy D4RL constructors return four-tuples. Instantiating their entry point
    directly avoids Gym 0.26's incompatible outer TimeLimit/checker wrappers.
    """

    def __init__(self, env, name, horizon, seed, reward_scale, reward_bias):
        self.env, self.name, self.horizon = env, name, horizon
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.reward_scale, self.reward_bias = reward_scale, reward_bias
        self.steps = 0
        self.seed = seed
        if hasattr(env, "seed"):
            env.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        self.steps = 0
        obs = self.env.reset()
        return obs if isinstance(obs, tuple) and len(obs) == 2 else (obs, {})

    def step(self, action):
        # Match WSRL's action rescaling; Adroit actions are passed through.
        if "binary" not in self.name:
            limit = 0.99999
            action = np.clip(action, -limit, limit) / limit
            low, high = self.action_space.low, self.action_space.high
            action = low + (action + 1.0) * 0.5 * (high - low)
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
        if "kitchen" in self.name and reward == 4:
            terminated = True
        if "binary" in self.name and info.get("goal_achieved", False):
            terminated = True
        self.steps += 1
        truncated = bool(truncated or (self.steps >= self.horizon and not terminated))
        reward = reward * self.reward_scale + self.reward_bias
        return obs, reward, bool(terminated), truncated, info

    def get_normalized_score(self, score):
        return self.env.get_normalized_score(score)

    def close(self):
        self.env.close()


def make_gym_env(env_name, reward_scale=1.0, reward_bias=0.0, seed=0):
    import importlib
    import gym

    # Register only the domain being used. Adroit is an optional dependency.
    if "binary" in env_name:
        importlib.import_module("mj_envs.hand_manipulation_suite")
    else:
        importlib.import_module("d4rl")
    spec = gym.spec(env_name)
    constructor = spec.entry_point
    if isinstance(constructor, str):
        module, name = constructor.split(":")
        constructor = getattr(importlib.import_module(module), name)
    env = constructor(**spec.kwargs)
    env.spec = spec
    if spec.max_episode_steps is None:
        raise ValueError(f"No episode horizon registered for {env_name}")
    return BenchmarkEnv(env, env_name, spec.max_episode_steps, seed, reward_scale, reward_bias)


def _determine_whether_sparse_reward(env_name):
    # return True if the environment is sparse-reward
    # determine if the env is sparse-reward or not
    if "antmaze" in env_name or env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]:
        is_sparse_reward = True
    elif (
        "halfcheetah" in env_name
        or "hopper" in env_name
        or "walker" in env_name
        or "kitchen" in env_name
    ):
        is_sparse_reward = False
    else:
        raise NotImplementedError

    return is_sparse_reward


# used to calculate the MC return for sparse-reward tasks.
# Assumes that the environment issues two reward values: reward_pos when the
# task is completed, and reward_neg at all the other steps.
ENV_REWARD_INFO = {
    "antmaze": {  # antmaze default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {  # adroit default is -1/0 reward
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    },
}


def _get_negative_reward(env_name, reward_scale, reward_bias):
    """
    Given an environment with sparse rewards (aka there's only two reward values,
    the goal reward when the task is done, or the step penalty otherwise).
    Args:
        env_name: the name of the environment
        reward_scale: the reward scale
        reward_bias: the reward bias. The reward_scale and reward_bias are not applied
            here to scale the reward, but to determine the correct negative reward value.

    NOTE: this function should only be called on sparse-reward environments
    """
    if "antmaze" in env_name:
        reward_neg = ENV_REWARD_INFO["antmaze"]["reward_neg"] * reward_scale + reward_bias
    elif env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
    ]:
        reward_neg = ENV_REWARD_INFO["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
    else:
        raise NotImplementedError(
            """
            If you want to try on a sparse reward env,
            please add the reward_neg value in the ENV_REWARD_INFO dict.
        """
        )

    return reward_neg


def calc_return_to_go(
    env_name,
    rewards,
    masks,
    gamma,
    reward_scale=None,
    reward_bias=None,
    infinite_horizon=False,
):
    """
    Calculat the Monte Carlo return to go given a list of reward for a single trajectory.
    Args:
        env_name: the name of the environment
        rewards: a list of rewards
        masks: a list of done masks
        gamma: the discount factor used to discount rewards
        reward_scale, reward_bias: the reward scale and bias used to determine
            the negative reward value for sparse-reward environments. If None,
            default to the unscaled environment rewards.
        infinite_horizon: whether the MDP has inifite horizion (and therefore infinite return to go)
    """
    if len(rewards) == 0:
        return np.array([])

    # process sparse-reward envs
    if reward_scale is None or reward_bias is None:
        # scale and bias not applied, but used to determien the negative reward value
        assert reward_scale is None and reward_bias is None  # both should be unset
        reward_scale = 1.0
        reward_bias = 0.0
    is_sparse_reward = _determine_whether_sparse_reward(env_name)
    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        # sum up the rewards backwards as the return to go
        return_to_go = [0] * len(rewards)
        prev_return = 0 if not infinite_horizon else float(rewards[-1] / (1 - gamma))
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (masks[-i - 1])
            prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)
