from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    image_shape: tuple[int, int, int] = (1, 84, 84)
    state_dim: int = 6
    action_dim: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    update_epochs: int = 4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOActorCritic(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int], state_dim: int, action_dim: int):
        super().__init__()
        c, h, w = image_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.cnn(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(cnn_out + state_dim, 512),
            nn.Tanh(),
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual_features = self.cnn(image)
        features = torch.cat([visual_features, state], dim=1)
        hidden = self.shared(features)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        return logits, value


class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.network = PPOActorCritic(config.image_shape, config.state_dim, config.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate, eps=1e-5)
        self.steps_done = 0

    def select_action(self, observation: dict[str, np.ndarray], evaluate: bool = False) -> int:
        image, state = self._obs_to_tensors(observation)
        with torch.no_grad():
            logits, _ = self.network(image, state)
            if evaluate:
                action = logits.argmax(dim=1)
            else:
                action = Categorical(logits=logits).sample()
        return int(action.item())

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        image, state = self._obs_to_tensors(observation)
        with torch.no_grad():
            logits, value = self.network(image, state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self.steps_done += 1
        return {
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }

    def value(self, observation: dict[str, np.ndarray]) -> float:
        image, state = self._obs_to_tensors(observation)
        with torch.no_grad():
            _, value = self.network(image, state)
        return float(value.item())

    def update(self, rollout: list[dict[str, Any]], next_value: float) -> dict[str, float]:
        cfg = self.config
        images = torch.as_tensor(
            np.stack([item["obs"]["image"] for item in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        states = torch.as_tensor(
            np.stack([item["obs"]["state"] for item in rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor([item["action"] for item in rollout], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([item["log_prob"] for item in rollout], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([item["reward"] for item in rollout], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item["done"] for item in rollout], dtype=torch.float32, device=self.device)
        values = torch.as_tensor([item["value"] for item in rollout], dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = 0.0
        for t in reversed(range(len(rollout))):
            next_non_terminal = 1.0 - dones[t]
            if t == len(rollout) - 1:
                next_values = torch.as_tensor(next_value, dtype=torch.float32, device=self.device)
            else:
                next_values = values[t + 1]
            delta = rewards[t] + cfg.gamma * next_values * next_non_terminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = len(rollout)
        indices = np.arange(batch_size)
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "updates": 0.0,
        }

        for _ in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, cfg.batch_size):
                batch_idx = torch.as_tensor(indices[start : start + cfg.batch_size], dtype=torch.long, device=self.device)

                logits, new_values = self.network(images[batch_idx], states[batch_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[batch_idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                policy_loss_1 = -advantages[batch_idx] * ratio
                policy_loss_2 = -advantages[batch_idx] * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                value_loss = 0.5 * (returns[batch_idx] - new_values).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float(approx_kl.item())
                stats["updates"] += 1.0

            if cfg.target_kl is not None and stats["updates"] > 0:
                mean_kl = stats["approx_kl"] / stats["updates"]
                if mean_kl > cfg.target_kl:
                    break

        if stats["updates"] > 0:
            for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
                stats[key] /= stats["updates"]
        return stats

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.steps_done = int(checkpoint.get("steps_done", 0))
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def _obs_to_tensors(self, observation: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(observation["image"][None, ...], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(observation["state"][None, ...], dtype=torch.float32, device=self.device)
        return image, state
