from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\AirSim\rl_drone_navigation")
OUT = ROOT / "tmp" / "deliverables" / "report" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

COMPARISON = ROOT / "experiments" / "airsimnh" / "validated_comparison_seed7_test_seed20007.csv"

RUNS = {
    "DQN Scratch": {
        "log": ROOT
        / "experiments"
        / "airsimnh"
        / "dqn"
        / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_log.csv",
        "trajectory": ROOT
        / "experiments"
        / "airsimnh"
        / "dqn"
        / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_trajectory.csv",
        "color": "#454B54",
    },
    "PPO Scratch": {
        "log": ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_log.csv",
        "trajectory": ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_trajectory.csv",
        "color": "#146C94",
    },
    "PPO Curriculum": {
        "log": ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_log.csv",
        "trajectory": ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot_validated_test_seed20007"
        / "results"
        / "evaluation_deterministic_trajectory.csv",
        "color": "#D88C00",
    },
}

TRAINING_LOGS = {
    "DQN Scratch": ROOT
    / "experiments"
    / "airsimnh"
    / "dqn"
    / "scratch_33m_45k_seed7_stable_v3_scratch"
    / "results"
    / "training_log.csv",
    "PPO Scratch": ROOT
    / "experiments"
    / "airsimnh"
    / "ppo"
    / "scratch_33m_45k_seed7_stable_v3_scratch"
    / "results"
    / "training_log.csv",
}

CURRICULUM_LOGS = [
    (
        0,
        ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "curriculum_stage01_10m_5k_seed7_stable_v3_stage2_pilot"
        / "results"
        / "training_log.csv",
    ),
    (
        5_000,
        ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "curriculum_stage02_23m_10k_seed7_stable_v3_stage2_pilot"
        / "results"
        / "training_log.csv",
    ),
    (
        15_000,
        ROOT
        / "experiments"
        / "airsimnh"
        / "ppo"
        / "curriculum_stage03_33m_30k_seed7_stable_v3_stage3_pilot"
        / "results"
        / "training_log.csv",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def make_outcome_chart() -> None:
    rows = [row for row in read_csv(COMPARISON) if row["policy_mode"] == "deterministic"]
    names = [row["method"] for row in rows]
    success = np.array([100.0 * float(row["success_rate"]) for row in rows])
    collision = np.array([100.0 * float(row["collision_rate"]) for row in rows])
    timeout = np.array([100.0 * float(row["timeout_rate"]) for row in rows])

    x = np.arange(len(names))
    width = 0.23
    fig, ax = plt.subplots(figsize=(6.8, 3.4), dpi=220)
    bars = [
        ax.bar(x - width, success, width, label="Success", color="#178F74"),
        ax.bar(x, collision, width, label="Collision", color="#C44536"),
        ax.bar(x + width, timeout, width, label="Timeout", color="#D88C00"),
    ]
    for group in bars:
        ax.bar_label(group, fmt="%.0f%%", padding=2, fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rate across 50 test episodes (%)")
    ax.set_xticks(x, ["DQN\nScratch", "PPO\nScratch", "PPO\nCurriculum"])
    ax.grid(axis="y", color="#D7DCE1", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.29))
    fig.tight_layout()
    fig.savefig(OUT / "final_outcomes.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def first_episode_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(path) if int(row["episode"]) == 1]
    x = [float(rows[0]["before_x"])] + [float(row["after_x"]) for row in rows]
    y = [float(rows[0]["before_y"])] + [float(row["after_y"]) for row in rows]
    return np.asarray(x), np.asarray(y)


def make_trajectory_chart() -> None:
    start = (85.413, -15.334)
    target = (117.756, -19.034)
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=220)
    for name, config in RUNS.items():
        x, y = first_episode_xy(config["trajectory"])
        ax.plot(x, y, linewidth=2.0, color=config["color"], label=name)
        ax.scatter(x[-1], y[-1], s=20, color=config["color"], zorder=4)
    ax.scatter(*start, marker="o", s=55, color="#178F74", edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter(*target, marker="*", s=120, color="#C44536", edgecolor="white", linewidth=0.8, zorder=5)
    ax.annotate("Start", start, xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.annotate("Target", target, xytext=(-38, -2), textcoords="offset points", fontsize=8)
    ax.set_xlabel("AirSim NED x (m)")
    ax.set_ylabel("AirSim NED y (m)")
    ax.grid(color="#D7DCE1", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUT / "representative_trajectories.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_action_chart() -> None:
    actions = ["forward", "left", "right", "up", "down", "hover"]
    colors = ["#178F74", "#146C94", "#63A6C6", "#D88C00", "#C44536", "#7A7F87"]
    names = list(RUNS)
    distributions: list[list[float]] = []
    for name in names:
        rows = read_csv(RUNS[name]["trajectory"])
        counts = {action: 0 for action in actions}
        for row in rows:
            action = row["action_name"]
            if action in counts:
                counts[action] += 1
        total = max(sum(counts.values()), 1)
        distributions.append([100.0 * counts[action] / total for action in actions])

    fig, ax = plt.subplots(figsize=(6.8, 3.1), dpi=220)
    left = np.zeros(len(names))
    y = np.arange(len(names))
    for action_index, (action, color) in enumerate(zip(actions, colors)):
        values = np.asarray([row[action_index] for row in distributions])
        bars = ax.barh(y, values, left=left, label=action.title(), color=color)
        for bar, value in zip(bars, values):
            if value >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if action in ("forward", "left", "down", "hover") else "#20252A",
                )
        left += values
    ax.set_yticks(y, ["DQN Scratch", "PPO Scratch", "PPO Curriculum"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of deterministic evaluation actions (%)")
    ax.grid(axis="x", color="#D7DCE1", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.48))
    fig.tight_layout()
    fig.savefig(OUT / "action_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_training(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    steps = np.asarray([float(row["global_step"]) for row in rows], dtype=float)
    success = np.asarray([float(row["success"]) for row in rows], dtype=float)
    return steps, success


def make_training_chart() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=220)
    for name, path in TRAINING_LOGS.items():
        x, success = read_training(path)
        ax.plot(
            x,
            moving_average(success),
            linewidth=2,
            label=name,
            color=RUNS[name]["color"],
        )

    curriculum_x: list[float] = []
    curriculum_success: list[float] = []
    for budget_offset, path in CURRICULUM_LOGS:
        x, success = read_training(path)
        relative = x - x[0]
        curriculum_x.extend((budget_offset + relative).tolist())
        curriculum_success.extend(success.tolist())
    ax.plot(
        np.asarray(curriculum_x),
        moving_average(np.asarray(curriculum_success)),
        linewidth=2,
        label="PPO Curriculum",
        color=RUNS["PPO Curriculum"]["color"],
    )
    ax.axvline(5_000, color="#AAB2BA", linestyle="--", linewidth=1)
    ax.axvline(15_000, color="#AAB2BA", linestyle="--", linewidth=1)
    ax.text(5_300, 0.04, "10 m -> 23 m", fontsize=8, color="#59636D")
    ax.text(15_300, 0.04, "23 m -> 33 m", fontsize=8, color="#59636D")
    ax.set_xlim(0, 45_000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Consumed environment interactions")
    ax.set_ylabel("20-episode moving success rate")
    ax.grid(color="#D7DCE1", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "training_success.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metrics() -> None:
    rows = [row for row in read_csv(COMPARISON) if row["policy_mode"] == "deterministic"]
    lines = []
    for row in rows:
        k = round(float(row["success_rate"]) * int(row["episodes"]))
        n = int(row["episodes"])
        p = k / n
        z = 1.96
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        lines.append(
            f"{row['method']}: successes={k}/{n}, Wilson95=[{100*(center-half):.1f}, {100*(center+half):.1f}]"
        )
    (OUT / "derived_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_outcome_chart()
    make_trajectory_chart()
    make_action_chart()
    make_training_chart()
    write_metrics()
