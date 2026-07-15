from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (ValueError, KeyError):
            values.append(np.nan)
    return np.array(values, dtype=np.float32)


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(csv_path: str | Path, output_path: str | Path):
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    episodes = _read_column(rows, "episode")
    reward = _read_column(rows, "reward")
    success = _read_column(rows, "success")
    collision = _read_column(rows, "collision")
    final_distance = _read_column(rows, "final_distance")
    epsilon = _read_column(rows, "epsilon")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(episodes, reward, alpha=0.35, label="episode")
    ma_reward = moving_average(reward)
    axes[0].plot(episodes[-len(ma_reward) :], ma_reward, label="moving avg")
    axes[0].set_title("Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    ma_success = moving_average(success)
    axes[1].plot(episodes[-len(ma_success) :], ma_success)
    axes[1].set_title("Success Rate (Moving Average)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)

    ma_collision = moving_average(collision)
    axes[2].plot(episodes[-len(ma_collision) :], ma_collision)
    axes[2].set_title("Collision Rate (Moving Average)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Collision")
    axes[2].set_ylim(-0.05, 1.05)

    axes[3].plot(episodes, final_distance, label="final distance")
    if not np.isnan(epsilon).all():
        axes[3].plot(episodes, epsilon, label="epsilon")
        axes[3].set_title("Final Distance and Exploration")
    else:
        axes[3].set_title("Final Distance")
    axes[3].set_xlabel("Episode")
    axes[3].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_training_curves("results/training_log.csv", "results/training_curves.png")
