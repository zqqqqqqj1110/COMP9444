from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep_checkpoints import (  # noqa: E402
    checkpoint_run_step,
    discover_checkpoints,
    rank_summaries,
)


class CheckpointSweepTests(unittest.TestCase):
    def test_discovers_only_training_candidates_for_each_algorithm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)
            for name in (
                "ppo_step_0002500.pt",
                "ppo_step_0005000.pt",
                "ppo_best.pt",
                "ppo_final.pt",
                "ppo_best_deterministic.pt",
                "dqn_step_0002500.pt",
                "dqn_final.pt",
                "dqn_best_deterministic.pt",
            ):
                (models_dir / name).touch()

            ppo_names = [
                path.name for path in discover_checkpoints(models_dir, "ppo")
            ]
            dqn_names = [
                path.name for path in discover_checkpoints(models_dir, "dqn")
            ]

        self.assertEqual(
            ppo_names,
            [
                "ppo_step_0002500.pt",
                "ppo_step_0005000.pt",
                "ppo_best.pt",
                "ppo_final.pt",
            ],
        )
        self.assertEqual(
            dqn_names,
            ["dqn_step_0002500.pt", "dqn_final.pt"],
        )

    def test_checkpoint_step_supports_dqn_and_ppo(self):
        self.assertEqual(
            checkpoint_run_step(Path("dqn_step_0030000.pt"), "dqn"),
            30_000,
        )
        self.assertEqual(
            checkpoint_run_step(Path("ppo_step_0045000.pt"), "ppo"),
            45_000,
        )
        self.assertIsNone(checkpoint_run_step(Path("ppo_final.pt"), "ppo"))

    def test_ranking_prioritizes_success_then_safety(self):
        rows = [
            {
                "checkpoint": "unsafe.pt",
                "success_rate": 1.0,
                "unsafe_rate": 0.2,
                "average_final_distance": 1.0,
                "average_steps": 40.0,
            },
            {
                "checkpoint": "safe.pt",
                "success_rate": 1.0,
                "unsafe_rate": 0.0,
                "average_final_distance": 2.0,
                "average_steps": 50.0,
            },
            {
                "checkpoint": "lower_success.pt",
                "success_rate": 0.9,
                "unsafe_rate": 0.0,
                "average_final_distance": 1.0,
                "average_steps": 30.0,
            },
        ]

        ranked = rank_summaries(rows)

        self.assertEqual(
            [row["checkpoint"] for row in ranked],
            ["safe.pt", "unsafe.pt", "lower_success.pt"],
        )
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
