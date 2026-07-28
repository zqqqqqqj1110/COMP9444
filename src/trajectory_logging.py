from __future__ import annotations

from typing import Any

import numpy as np


ACTION_NAMES = ("forward", "left", "right", "up", "down", "hover")
ACTION_COUNT_FIELDS = tuple(f"action_{index}_count" for index in range(len(ACTION_NAMES)))
ACTION_PROBABILITY_FIELDS = tuple(
    f"action_{index}_probability" for index in range(len(ACTION_NAMES))
)
REWARD_COMPONENT_FIELDS = (
    "step_penalty",
    "progress_reward",
    "altitude_hold_penalty",
    "altitude_margin_penalty",
    "goal_reward",
    "collision_penalty",
    "altitude_penalty",
    "timeout_penalty",
)


def action_count_values(action_counts: np.ndarray) -> dict[str, int]:
    return {
        field: int(action_counts[index])
        for index, field in enumerate(ACTION_COUNT_FIELDS)
    }


def action_probability_values(probabilities: np.ndarray | None) -> dict[str, float]:
    if probabilities is None:
        return {field: float("nan") for field in ACTION_PROBABILITY_FIELDS}
    return {
        field: float(probabilities[index])
        for index, field in enumerate(ACTION_PROBABILITY_FIELDS)
    }


def reward_breakdown(info: dict[str, Any], reward: float) -> dict[str, float | str]:
    total_reward = float(reward)
    if total_reward > 1e-9:
        reward_type = "reward"
    elif total_reward < -1e-9:
        reward_type = "penalty"
    else:
        reward_type = "neutral"

    values: dict[str, float | str] = {
        "reward": total_reward,
        "reward_type": reward_type,
        "positive_reward": max(total_reward, 0.0),
        "penalty_amount": max(-total_reward, 0.0),
    }
    for field in REWARD_COMPONENT_FIELDS:
        values[field] = float(info.get(field, 0.0))
    return values
