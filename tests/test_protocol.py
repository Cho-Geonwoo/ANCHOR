import unittest

from train import parse_args
from wsrl.benchmark import ENVS, METHODS, TASKS, agent_config, kl_weight, protocol


class ProtocolTests(unittest.TestCase):
    def test_paper_hyperparameters(self):
        expected = [
            (1.0, 1.0, 160000, 1),
            (1.0, 1.0, 160000, 1),
            (0.0, 0.2, 40000, 16),
            (0.0, 1.0, 80000, 1),
            (0.0, 0.2, 80000, 1),
            (0.0, 0.4, 80000, 1),
        ]
        for env, values in zip(ENVS, expected):
            task = TASKS[env]
            self.assertEqual((task.alpha_on, task.beta0, task.kl_steps, task.anchor_utd), values)
            cfg = agent_config("anchor", env)
            self.assertFalse(cfg["cql_autotune_alpha"])
            self.assertEqual(cfg["cql_alpha"], task.alpha_off)
            self.assertEqual(cfg["online_cql_alpha"], task.alpha_on)

    def test_linear_schedule_includes_warmup(self):
        self.assertEqual(kl_weight(0.4, 0, 80000), 0.4)
        self.assertAlmostEqual(kl_weight(0.4, 40000, 80000), 0.2)
        self.assertEqual(kl_weight(0.4, 80000, 80000), 0.0)
        self.assertEqual(kl_weight(0.4, 100000, 80000), 0.0)

    def test_all_method_task_configs_and_cli_defaults(self):
        for method in METHODS:
            for env in ENVS:
                spec = protocol(method, env)
                args = parse_args(["--method", method, "--env", env])
                for key in ("offline_steps", "online_steps", "utd", "warmup_steps", "batch_size"):
                    self.assertEqual(getattr(args, key), spec[key])
                cfg = agent_config(method, env)
                self.assertEqual(cfg["critic_ensemble_size"], 10)
                self.assertEqual(cfg["critic_subsample_size"], 2)
                self.assertTrue(cfg["policy_network_kwargs"]["use_layer_norm"])
                self.assertTrue(cfg["critic_network_kwargs"]["use_layer_norm"])


if __name__ == "__main__":
    unittest.main()
