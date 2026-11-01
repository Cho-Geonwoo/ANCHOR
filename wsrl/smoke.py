"""Tiny synthetic dynamics for tests, never a benchmark result."""

import gym
import numpy as np


class SmokeEnv:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self):
        self.t = 0
        self.state = self.rng.normal(0, 0.1, 5).astype(np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.t += 1
        self.state[:2] += np.asarray(action) * 0.1
        reward = float(np.linalg.norm(self.state[:2]) < 0.3)
        return self.state.copy(), reward, False, self.t >= 4, {"goal_achieved": bool(reward)}

    def close(self):
        pass


def dataset(seed, size=64):
    rng = np.random.default_rng(seed)
    observations = rng.normal(0, 0.1, (size, 5)).astype(np.float32)
    return dict(
        observations=observations,
        next_observations=observations + 0.01,
        actions=rng.uniform(-0.8, 0.8, (size, 2)).astype(np.float32),
        rewards=np.zeros(size, np.float32),
        masks=np.ones(size, np.float32),
        dones=np.zeros(size, np.float32),
        mc_returns=np.zeros(size, np.float32),
    )
