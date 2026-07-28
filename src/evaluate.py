from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from dqn_agent import DQNAgent, DQNConfig
from experiment_paths import (
    default_model_path,
    ensure_experiment_dirs,
    print_experiment_paths,
    resolve_experiment_paths,
)
from ppo_agent import PPOAgent, PPOConfig
from trajectory_logging import (
    ACTION_COUNT_FIELDS,
    ACTION_NAMES,
    ACTION_PROBABILITY_FIELDS,
    REWARD_COMPONENT_FIELDS,
    action_count_values,
    action_probability_values,
    reward_breakdown,
)


POLICY_MODES = ("deterministic", "stochastic")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained drone navigation agent in AirSim.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--scenario", type=str, default="blocks", help="Scenario name used for experiment outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder used during training.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--policy-mode",
        choices=["deterministic", "stochastic", "both"],
        default="deterministic",
        help="PPO action selection mode. 'both' writes separate outputs for both policies.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-x", type=float, default=20.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=-3.0)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for evaluation files.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Optional override for model lookup.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agent(algorithm: str, model_path: Path):
    if algorithm == "dqn":
        agent = DQNAgent(DQNConfig())
    elif algorithm == "ppo":
        agent = PPOAgent(PPOConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    agent.load(model_path)
    return agent


def evaluation_episode_fields() -> list[str]:
    return [
        "policy_mode",
        "episode",
        "reward",
        "steps",
        "success",
        "collision",
        "out_of_altitude",
        "timeout",
        "final_distance",
        "final_x",
        "final_y",
        "final_z",
        "path_length_m",
        "min_depth_m",
        "dominant_action",
        "dominant_action_name",
        "dominant_action_fraction",
        *ACTION_COUNT_FIELDS,
    ]


def evaluation_trajectory_fields() -> list[str]:
    return [
        "policy_mode",
        "episode",
        "step",
        "action",
        "action_name",
        "selected_action_probability",
        *ACTION_PROBABILITY_FIELDS,
        "before_x",
        "before_y",
        "before_z",
        "after_x",
        "after_y",
        "after_z",
        "distance_before",
        "distance_after",
        "reward",
        "reward_type",
        "positive_reward",
        "penalty_amount",
        *REWARD_COMPONENT_FIELDS,
        "cumulative_episode_reward",
        "success",
        "collision",
        "out_of_altitude",
        "timeout",
        "done",
        "collision_object",
        "min_depth_m",
    ]


def evaluate_policy(
    env: AirSimDroneEnv,
    agent: DQNAgent | PPOAgent,
    algorithm: str,
    policy_mode: str,
    episodes: int,
    max_steps: int,
    seed: int,
    collect_trajectory: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"Unsupported policy mode: {policy_mode}")
    if algorithm == "dqn" and policy_mode != "deterministic":
        raise ValueError("DQN evaluation supports deterministic mode only.")

    set_seed(seed)
    episode_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    deterministic = policy_mode == "deterministic"

    for episode in range(1, episodes + 1):
        observation, current_info = env.reset(seed=seed + episode - 1)
        total_reward = 0.0
        info = current_info
        action_counts = np.zeros(agent.config.action_dim, dtype=np.int64)
        terminated = False
        truncated = False

        for step in range(1, max_steps + 1):
            before_position = current_info.get("position", (np.nan, np.nan, np.nan))
            distance_before = current_info.get("distance_to_target", np.nan)
            probabilities = (
                agent.action_probabilities(observation)
                if hasattr(agent, "action_probabilities")
                else None
            )
            action = agent.select_action(observation, evaluate=deterministic)
            action_counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            after_position = info.get("position", (np.nan, np.nan, np.nan))

            if collect_trajectory:
                trajectory_rows.append(
                    {
                        "policy_mode": policy_mode,
                        "episode": episode,
                        "step": step,
                        "action": action,
                        "action_name": ACTION_NAMES[action],
                        "selected_action_probability": (
                            float(probabilities[action])
                            if probabilities is not None
                            else np.nan
                        ),
                        **action_probability_values(probabilities),
                        "before_x": before_position[0],
                        "before_y": before_position[1],
                        "before_z": before_position[2],
                        "after_x": after_position[0],
                        "after_y": after_position[1],
                        "after_z": after_position[2],
                        "distance_before": distance_before,
                        "distance_after": info.get("distance_to_target", np.nan),
                        **reward_breakdown(info, reward),
                        "cumulative_episode_reward": total_reward,
                        "success": int(info.get("success", False)),
                        "collision": int(info.get("collision", False)),
                        "out_of_altitude": int(info.get("out_of_altitude", False)),
                        "timeout": int(bool(truncated and not terminated)),
                        "done": int(done),
                        "collision_object": info.get("collision_object", ""),
                        "min_depth_m": info.get("episode_min_depth_m", np.nan),
                    }
                )
            current_info = info
            if done:
                break

        position = info.get("position", (np.nan, np.nan, np.nan))
        dominant_action = int(np.argmax(action_counts))
        observed_steps = max(int(action_counts.sum()), 1)
        episode_rows.append(
            {
                "policy_mode": policy_mode,
                "episode": episode,
                "reward": total_reward,
                "steps": info.get("steps", observed_steps),
                "success": int(info.get("success", False)),
                "collision": int(info.get("collision", False)),
                "out_of_altitude": int(info.get("out_of_altitude", False)),
                "timeout": int(bool(truncated and not terminated)),
                "final_distance": info.get("distance_to_target", np.nan),
                "final_x": position[0],
                "final_y": position[1],
                "final_z": position[2],
                "path_length_m": info.get("path_length_m", np.nan),
                "min_depth_m": info.get("episode_min_depth_m", np.nan),
                "dominant_action": dominant_action,
                "dominant_action_name": ACTION_NAMES[dominant_action],
                "dominant_action_fraction": float(
                    action_counts[dominant_action] / observed_steps
                ),
                **action_count_values(action_counts),
            }
        )

    return episode_rows, trajectory_rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | str]:
    unsafe_rate = float(
        np.mean(
            [
                bool(row["collision"] or row["out_of_altitude"])
                for row in rows
            ]
        )
    )
    return {
        "policy_mode": str(rows[0]["policy_mode"]),
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
        "altitude_violation_rate": float(
            np.mean([row["out_of_altitude"] for row in rows])
        ),
        "unsafe_rate": unsafe_rate,
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "average_reward": float(np.mean([row["reward"] for row in rows])),
        "average_steps": float(np.mean([row["steps"] for row in rows])),
        "average_final_distance": float(
            np.mean([row["final_distance"] for row in rows])
        ),
        "average_path_length_m": float(
            np.mean([row["path_length_m"] for row in rows])
        ),
        "average_min_depth_m": float(np.mean([row["min_depth_m"] for row in rows])),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mode_outputs(
    results_dir: Path,
    policy_mode: str,
    episode_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
):
    log_path = results_dir / f"evaluation_{policy_mode}_log.csv"
    trajectory_path = results_dir / f"evaluation_{policy_mode}_trajectory.csv"
    write_csv(log_path, evaluation_episode_fields(), episode_rows)
    write_csv(trajectory_path, evaluation_trajectory_fields(), trajectory_rows)

    # Keep deterministic aliases for existing plotting and summary scripts.
    if policy_mode == "deterministic":
        write_csv(results_dir / "evaluation_log.csv", evaluation_episode_fields(), episode_rows)
        write_csv(
            results_dir / "evaluation_trajectory.csv",
            evaluation_trajectory_fields(),
            trajectory_rows,
        )
    return log_path, trajectory_path


def print_summary(summary: dict[str, float | int | str]):
    print(f"{str(summary['policy_mode']).title()} evaluation")
    print(f"  Success rate: {float(summary['success_rate']):.2%}")
    print(f"  Collision rate: {float(summary['collision_rate']):.2%}")
    print(
        f"  Altitude violation rate: "
        f"{float(summary['altitude_violation_rate']):.2%}"
    )
    print(f"  Unsafe rate: {float(summary['unsafe_rate']):.2%}")
    print(f"  Timeout rate: {float(summary['timeout_rate']):.2%}")
    print(f"  Average reward: {float(summary['average_reward']):.2f}")
    print(f"  Average steps: {float(summary['average_steps']):.1f}")
    print(f"  Average final distance: {float(summary['average_final_distance']):.2f} m")


def main():
    args = parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max-steps must be positive.")
    if args.algorithm == "dqn" and args.policy_mode != "deterministic":
        raise ValueError("DQN supports deterministic evaluation only.")

    paths = resolve_experiment_paths(
        scenario=args.scenario,
        algorithm=args.algorithm,
        output_root=args.output_root,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        run_name=args.run_name,
    )
    ensure_experiment_dirs(paths)
    print_experiment_paths(paths)

    model_path = args.model if args.model is not None else default_model_path(paths)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first or pass --model with a checkpoint path."
        )

    env = AirSimDroneEnv(
        DroneEnvConfig(
            max_steps=args.max_steps,
            target_position=(args.target_x, args.target_y, args.target_z),
            start_position=(args.start_x, args.start_y, args.start_z),
        )
    )
    agent = load_agent(args.algorithm, model_path)
    modes = POLICY_MODES if args.policy_mode == "both" else (args.policy_mode,)
    summaries: list[dict[str, float | int | str]] = []

    try:
        for mode in modes:
            episode_rows, trajectory_rows = evaluate_policy(
                env=env,
                agent=agent,
                algorithm=args.algorithm,
                policy_mode=mode,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                collect_trajectory=True,
            )
            log_path, trajectory_path = write_mode_outputs(
                paths.results_dir,
                mode,
                episode_rows,
                trajectory_rows,
            )
            summary = summarize_rows(episode_rows)
            summaries.append(summary)
            print("")
            print_summary(summary)
            print(f"  Episode log: {log_path}")
            print(f"  Action/reward trajectory: {trajectory_path}")
    finally:
        env.close()

    comparison_path = paths.results_dir / "evaluation_mode_comparison.csv"
    write_csv(comparison_path, list(summaries[0].keys()), summaries)
    summary_path = paths.results_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "model": str(model_path.resolve()),
                "seed": args.seed,
                "requested_policy_mode": args.policy_mode,
                "modes": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("")
    print(f"Model: {model_path}")
    print(f"Mode comparison: {comparison_path}")
    print(f"Evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
