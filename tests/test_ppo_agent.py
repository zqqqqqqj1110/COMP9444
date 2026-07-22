from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppo_agent import PPOAgent, PPOConfig  # noqa: E402


class PPOAgentTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(7)
        torch.manual_seed(7)

    @staticmethod
    def observation() -> dict[str, np.ndarray]:
        return {
            "image": np.random.random((1, 84, 84)).astype(np.float32),
            "state": np.array([0.6, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def test_feature_normalization_prevents_tanh_saturation(self):
        agent = PPOAgent(PPOConfig(device="cpu"))
        with torch.no_grad():
            agent.network.shared[0].weight.mul_(50.0)

        observations = [self.observation() for _ in range(16)]
        images = torch.as_tensor(np.stack([item["image"] for item in observations]))
        states = torch.as_tensor(np.stack([item["state"] for item in observations]))
        diagnostics = agent.network.diagnostics(images, states)

        self.assertLess(diagnostics["activation_saturation"], 0.03)
        self.assertLess(diagnostics["activation_abs_mean"], 0.8)

    def test_legacy_checkpoint_keeps_legacy_feature_behavior(self):
        legacy = PPOAgent(
            PPOConfig(
                device="cpu",
                normalize_shared_features=False,
                reward_scale=1.0,
                value_loss_type="mse",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "legacy.pt"
            legacy.save(checkpoint)

            loaded = PPOAgent(PPOConfig(device="cpu"))
            loaded.load(checkpoint)
            checkpoint_is_stable = PPOAgent.checkpoint_uses_stable_features(checkpoint)

        self.assertFalse(loaded.config.normalize_shared_features)
        self.assertFalse(loaded.network.normalize_shared_features)
        self.assertFalse(checkpoint_is_stable)

    def test_large_terminal_rewards_produce_finite_stable_update(self):
        agent = PPOAgent(
            PPOConfig(
                device="cpu",
                batch_size=16,
                update_epochs=2,
                reward_scale=0.1,
                value_loss_type="huber",
            )
        )
        rollout = []
        for step in range(32):
            obs = self.observation()
            action_data = agent.act(obs)
            done = (step + 1) % 8 == 0
            reward = -100.0 if done else float(np.random.uniform(-1.0, 1.0))
            rollout.append(
                {
                    "obs": obs,
                    "action": action_data["action"],
                    "log_prob": action_data["log_prob"],
                    "value": action_data["value"],
                    "reward": reward,
                    "done": done,
                }
            )

        stats = agent.update(rollout, next_value=0.0)

        for key, value in stats.items():
            self.assertTrue(math.isfinite(float(value)), key)
        self.assertLess(stats["activation_saturation"], 0.03)
        self.assertGreater(stats["entropy"], 1.0)


if __name__ == "__main__":
    unittest.main()
