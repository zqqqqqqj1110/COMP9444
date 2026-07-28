from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import evaluate_policy, write_mode_outputs  # noqa: E402


class FakeAgent:
    def __init__(self):
        self.config = SimpleNamespace(action_dim=6)
        self._actions = iter((2, 1))

    def action_probabilities(self, observation):
        del observation
        return np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2], dtype=np.float32)

    def select_action(self, observation, evaluate=False):
        del observation, evaluate
        return next(self._actions)


class FakeEnv:
    def __init__(self):
        self.step_count = 0

    @staticmethod
    def observation():
        return {
            "image": np.zeros((1, 84, 84), dtype=np.float32),
            "state": np.zeros(6, dtype=np.float32),
        }

    def reset(self, seed=None):
        del seed
        self.step_count = 0
        return self.observation(), {
            "position": (0.0, 0.0, -3.0),
            "distance_to_target": 2.0,
        }

    def step(self, action):
        self.step_count += 1
        success = self.step_count == 2
        progress_reward = 0.5
        goal_reward = 100.0 if success else 0.0
        reward = -0.05 + progress_reward + goal_reward
        info = {
            "position": (float(self.step_count), float(action), -3.0),
            "distance_to_target": 2.0 - self.step_count,
            "steps": self.step_count,
            "success": success,
            "collision": False,
            "out_of_altitude": False,
            "path_length_m": float(self.step_count),
            "episode_min_depth_m": 3.0,
            "step_penalty": -0.05,
            "progress_reward": progress_reward,
            "altitude_hold_penalty": 0.0,
            "altitude_margin_penalty": 0.0,
            "goal_reward": goal_reward,
            "collision_penalty": 0.0,
            "altitude_penalty": 0.0,
            "timeout_penalty": 0.0,
            "collision_object": "",
        }
        return self.observation(), reward, success, False, info


class EvaluationLoggingTests(unittest.TestCase):
    def test_complete_action_counts_positions_and_reward_breakdown(self):
        episodes, trajectory = evaluate_policy(
            env=FakeEnv(),
            agent=FakeAgent(),
            algorithm="ppo",
            policy_mode="deterministic",
            episodes=1,
            max_steps=5,
            seed=7,
            collect_trajectory=True,
        )

        self.assertEqual(episodes[0]["action_1_count"], 1)
        self.assertEqual(episodes[0]["action_2_count"], 1)
        self.assertEqual(sum(episodes[0][f"action_{index}_count"] for index in range(6)), 2)
        self.assertEqual(trajectory[0]["action_name"], "right")
        self.assertEqual(trajectory[1]["action_name"], "left")
        self.assertEqual(trajectory[1]["before_x"], 1.0)
        self.assertEqual(trajectory[1]["reward_type"], "reward")
        self.assertAlmostEqual(trajectory[1]["goal_reward"], 100.0)
        self.assertAlmostEqual(trajectory[1]["positive_reward"], 100.45)

    def test_deterministic_outputs_keep_compatibility_aliases(self):
        episodes, trajectory = evaluate_policy(
            env=FakeEnv(),
            agent=FakeAgent(),
            algorithm="ppo",
            policy_mode="deterministic",
            episodes=1,
            max_steps=5,
            seed=7,
            collect_trajectory=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            write_mode_outputs(results_dir, "deterministic", episodes, trajectory)
            expected = (
                "evaluation_deterministic_log.csv",
                "evaluation_deterministic_trajectory.csv",
                "evaluation_log.csv",
                "evaluation_trajectory.csv",
            )
            for name in expected:
                self.assertTrue((results_dir / name).is_file(), name)
            with (results_dir / "evaluation_trajectory.csv").open(
                newline="",
                encoding="utf-8",
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("penalty_amount", rows[0])


if __name__ == "__main__":
    unittest.main()
