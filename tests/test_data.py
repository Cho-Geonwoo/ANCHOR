import unittest
import gym
import numpy as np
from wsrl.data.replay_buffer import ReplayBufferMC
from wsrl.envs.env_common import BenchmarkEnv, calc_return_to_go


class DataTests(unittest.TestCase):
    def test_sparse_mc_returns_and_incomplete_episode_exclusion(self):
        np.testing.assert_allclose(
            calc_return_to_go(
                "door-binary-v0", [-5.0, -5.0], [1.0, 1.0], 0.99, reward_scale=10.0, reward_bias=5.0
            ),
            [-500.0, -500.0],
        )
        space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        replay = ReplayBufferMC(
            space,
            space,
            20,
            seed=10,
            discount=0.99,
            env_name="door-binary-v0",
            reward_scale=10.0,
            reward_bias=5.0,
        )
        transition = dict(
            observations=np.zeros(2),
            next_observations=np.zeros(2),
            actions=np.zeros(2),
            rewards=-5.0,
            masks=1.0,
            dones=0.0,
        )
        replay.insert(dict(transition))
        self.assertEqual(replay._allow_idxs, [])
        replay.insert(dict(transition, dones=1.0))
        np.testing.assert_allclose(replay.sample(8)["mc_returns"], -500.0)

    def test_time_limit_bootstraps_but_success_terminates(self):
        class Legacy:
            observation_space = action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

            def seed(self, seed):
                pass

            def reset(self):
                return np.zeros(2)

            def step(self, action):
                return np.zeros(2), 0.0, False, {"TimeLimit.truncated": False}

            def close(self):
                pass

        env = BenchmarkEnv(Legacy(), "antmaze-large-play-v2", 2, 1, 1.0, 0.0)
        env.reset()
        self.assertEqual(env.step(np.zeros(2))[2:4], (False, False))
        self.assertEqual(env.step(np.zeros(2))[2:4], (False, True))


if __name__ == "__main__":
    unittest.main()
