import csv
import json
import tempfile
import unittest
from pathlib import Path

from wsrl.plotting import load_runs


class PlotInputTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "anchor/kitchen-partial-v0/seed10"
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps(
                dict(
                    method="anchor",
                    env="kitchen-partial-v0",
                    seed=10,
                    status="complete",
                    data_kind="synthetic_fixture",
                    arguments=dict(online_steps=10, eval_interval=5, eval_episodes=2),
                )
            )
        )
        with (run / "evaluations.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["phase", "offline_step", "online_step", "success_rate", "return", "num_episodes"]
            )
            writer.writerows([["online", 2, step, 0.5, 1.0, 2] for step in (0, 5, 10)])
        with (run / "episodes.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "phase",
                    "offline_step",
                    "online_step",
                    "episode",
                    "return",
                    "length",
                    "success_rate",
                ]
            )
            writer.writerows(
                [
                    ["online", 2, step, seed, 1.0, 4, float(seed)]
                    for step in (0, 5, 10)
                    for seed in (0, 1)
                ]
            )
        return run

    def test_synthetic_is_rejected_and_complete_fixture_can_be_loaded_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Synthetic run"):
                load_runs(root, ["anchor"], ["kitchen-partial-v0"], [10], 10)
            result = load_runs(root, ["anchor"], ["kitchen-partial-v0"], [10], 10, True)
            self.assertEqual(result[1], [0, 5, 10])
            self.assertEqual(len(result[3]), 6)

    def test_missing_evaluation_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = self.fixture(root)
            path = run / "evaluations.csv"
            lines = path.read_text().splitlines(True)
            path.write_text("".join(lines[:-1]))
            with self.assertRaisesRegex(ValueError, "off-grid evaluations"):
                load_runs(root, ["anchor"], ["kitchen-partial-v0"], [10], 10, True)


if __name__ == "__main__":
    unittest.main()
