from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Transition(NamedTuple):
    image: np.ndarray
    state: np.ndarray
    action: int
    reward: float
    next_image: np.ndarray
    next_state: np.ndarray
    done: bool


@dataclass
class DQNConfig:
    image_shape: tuple[int, int, int] = (1, 84, 84)
    state_dim: int = 6
    action_dim: int = 6
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    replay_capacity: int = 50_000
    min_replay_size: int = 1_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNNetwork(nn.Module):
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

        self.head = nn.Sequential(
            nn.Linear(cnn_out + state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.cnn(image)
        combined = torch.cat([features, state], dim=1)
        return self.head(combined)


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.policy_net = DQNNetwork(config.image_shape, config.state_dim, config.action_dim).to(self.device)
        self.target_net = DQNNetwork(config.image_shape, config.state_dim, config.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.steps_done = 0

    def epsilon(self) -> float:
        cfg = self.config
        fraction = min(1.0, self.steps_done / cfg.epsilon_decay_steps)
        return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, observation: dict[str, np.ndarray], evaluate: bool = False) -> int:
        if (not evaluate) and random.random() < self.epsilon():
            action = random.randrange(self.config.action_dim)
        else:
            image, state = self._obs_to_tensors(observation)
            with torch.no_grad():
                q_values = self.policy_net(image, state)
            action = int(q_values.argmax(dim=1).item())

        if not evaluate:
            self.steps_done += 1
        return action

    def remember(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ):
        self.replay.push(
            Transition(
                image=observation["image"].copy(),
                state=observation["state"].copy(),
                action=int(action),
                reward=float(reward),
                next_image=next_observation["image"].copy(),
                next_state=next_observation["state"].copy(),
                done=bool(done),
            )
        )

    def optimize(self) -> float | None:
        cfg = self.config
        if len(self.replay) < max(cfg.min_replay_size, cfg.batch_size):
            return None

        batch = self.replay.sample(cfg.batch_size)
        image = torch.as_tensor(np.stack([t.image for t in batch]), dtype=torch.float32, device=self.device)
        state = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        reward = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_image = torch.as_tensor(np.stack([t.next_image for t in batch]), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        done = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(image, state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_image, next_state).max(dim=1, keepdim=True).values
            target = reward + cfg.gamma * next_q * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.steps_done % cfg.target_update_interval == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = int(checkpoint.get("steps_done", 0))
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def _obs_to_tensors(self, observation: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.as_tensor(observation["image"][None, ...], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(observation["state"][None, ...], dtype=torch.float32, device=self.device)
        return image, state
