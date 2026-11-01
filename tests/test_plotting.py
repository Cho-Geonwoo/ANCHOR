import unittest
from wsrl.benchmark import ENVS
from wsrl.plotting import aggregate, summarize


class PlottingTests(unittest.TestCase):
    def test_sample_se_and_no_uncertainty_for_one_seed(self):
        stats = summarize([0.0, 1.0])
        self.assertEqual(stats["mean"], 0.5)
        self.assertAlmostEqual(stats["standard_error"], 0.5)
        self.assertIsNone(summarize([0.7])["standard_error"])

    def test_aggregate_weights_tasks_equally_before_seeds(self):
        series = {
            ("anchor", env, seed): [i / 10 + seed / 100]
            for i, env in enumerate(ENVS)
            for seed in (10, 20)
        }
        per_seed, curves = aggregate(series, [0], ["anchor"], ENVS, [10, 20])
        avg = next(r for r in curves if r["env"] == "aggregate")
        self.assertAlmostEqual(avg["mean"], 0.4)
        self.assertAlmostEqual(avg["standard_error"], 0.05)
        self.assertEqual(len(per_seed), 14)


if __name__ == "__main__":
    unittest.main()
