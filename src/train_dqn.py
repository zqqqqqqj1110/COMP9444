from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from dqn_agent import DQNAgent, DQNConfig
from experiment_paths import default_model_path, ensure_experiment_dirs, print_experiment_paths, resolve_experiment_paths, write_metadata
from plot_results import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN drone navigation agent in AirSim.")
    parser.add_argument("--scenario", type=str, default="blocks", help="Scenario name used for experiment outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder below the DQN experiment.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Stop after this many environment interactions instead of using the episode limit.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-x", type=float, default=20.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=-3.0)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for result files.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Optional override for model checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Save after each N environment interactions; 0 disables step checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.total_steps is not None and args.total_steps <= 0:
        raise ValueError("--total-steps must be positive when provided.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps cannot be negative.")
    set_seed(args.seed)
    training_started_at = datetime.now()
    training_started_perf = time.perf_counter()
    env_config = DroneEnvConfig(
        max_steps=args.max_steps,
        target_position=(args.target_x, args.target_y, args.target_z),
        start_position=(args.start_x, args.start_y, args.start_z),
    )
    paths = resolve_experiment_paths(
        scenario=args.scenario,
        algorithm="dqn",
        output_root=args.output_root,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        run_name=args.run_name,
    )
    ensure_experiment_dirs(paths)
    print_experiment_paths(paths)
    write_metadata(
        paths,
        {
            "script": "train_dqn.py",
            "target_position": [args.target_x, args.target_y, args.target_z],
            "start_position": [args.start_x, args.start_y, args.start_z],
            "episodes": args.episodes,
            "total_steps": args.total_steps,
            "stop_condition": "total_steps" if args.total_steps is not None else "episodes",
            "max_steps": args.max_steps,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "seed": args.seed,
            "reward_config": {
                "step_penalty": env_config.step_penalty,
                "distance_reward_scale": env_config.distance_reward_scale,
                "goal_reward": env_config.goal_reward,
                "collision_penalty": env_config.collision_penalty,
                "altitude_penalty": env_config.altitude_penalty,
                "altitude_hold_penalty_scale": env_config.altitude_hold_penalty_scale,
                "altitude_margin_m": env_config.altitude_margin_m,
                "altitude_margin_penalty_scale": env_config.altitude_margin_penalty_scale,
                "timeout_penalty": env_config.timeout_penalty,
            },
        },
    )

    agent_config = DQNConfig()

    env = AirSimDroneEnv(env_config)
    agent = DQNAgent(agent_config)

    log_path = paths.results_dir / "training_log.csv"
    fieldnames = [
        "episode",
        "global_step",
        "reward",
        "steps",
        "success",
        "collision",
        "out_of_altitude",
        "final_distance",
        "epsilon",
        "mean_loss",
        "timeout",
        "final_x",
        "final_y",
        "final_z",
        "path_length_m",
        "min_depth_m",
        "dominant_action",
        "dominant_action_fraction",
    ]

    use_step_budget = args.total_steps is not None
    run_steps = 0
    episode = 0
    progress = tqdm(
        total=args.total_steps if use_step_budget else args.episodes,
        desc="DQN steps" if use_step_budget else "DQN episodes",
        unit="step" if use_step_budget else "episode",
    )

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            while (run_steps < args.total_steps) if use_step_budget else (episode < args.episodes):
                episode += 1
                obs, _ = env.reset()
                total_reward = 0.0
                losses: list[float] = []
                info = {}
                action_counts = np.zeros(agent.config.action_dim, dtype=np.int64)
                terminated = False
                truncated = False

                for _ in range(args.max_steps):
                    if use_step_budget and run_steps >= args.total_steps:
                        break
                    action = agent.select_action(obs)
                    action_counts[action] += 1
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    agent.remember(obs, action, reward, next_obs, done)
                    loss = agent.optimize()
                    if loss is not None:
                        losses.append(loss)

                    obs = next_obs
                    total_reward += reward
                    run_steps += 1
                    if use_step_budget:
                        progress.update(1)

                    if args.checkpoint_every_steps > 0 and run_steps % args.checkpoint_every_steps == 0:
                        agent.save(paths.models_dir / f"dqn_step_{run_steps:07d}.pt")
                    if done:
                        break

                position = info.get("position", (np.nan, np.nan, np.nan))
                dominant_action = int(np.argmax(action_counts))
                observed_steps = max(int(action_counts.sum()), 1)
                row = {
                    "episode": episode,
                    "global_step": run_steps,
                    "reward": total_reward,
                    "steps": info.get("steps", args.max_steps),
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "out_of_altitude": int(info.get("out_of_altitude", False)),
                    "final_distance": info.get("distance_to_target", np.nan),
                    "epsilon": agent.epsilon(),
                    "mean_loss": float(np.mean(losses)) if losses else np.nan,
                    "timeout": int(bool(truncated and not terminated)),
                    "final_x": position[0],
                    "final_y": position[1],
                    "final_z": position[2],
                    "path_length_m": info.get("path_length_m", np.nan),
                    "min_depth_m": info.get("episode_min_depth_m", np.nan),
                    "dominant_action": dominant_action,
                    "dominant_action_fraction": float(action_counts[dominant_action] / observed_steps),
                }
                writer.writerow(row)
                f.flush()

                if not use_step_budget:
                    progress.update(1)

                if (not use_step_budget) and episode % args.checkpoint_every == 0:
                    agent.save(paths.models_dir / f"dqn_episode_{episode:04d}.pt")
                    plot_training_curves(log_path, paths.results_dir / "training_curves.png")

        finally:
            progress.close()
            env.close()

    final_path = default_model_path(paths)
    agent.save(final_path)
    plot_training_curves(log_path, paths.results_dir / "training_curves.png")
    training_completed_at = datetime.now()
    elapsed_seconds = time.perf_counter() - training_started_perf
    summary_path = paths.results_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scenario": paths.scenario,
                "algorithm": paths.algorithm,
                "run_name": paths.run_name,
                "status": "completed",
                "started_at": training_started_at.isoformat(timespec="seconds"),
                "completed_at": training_completed_at.isoformat(timespec="seconds"),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "elapsed_hours": round(elapsed_seconds / 3600.0, 6),
                "completed_episodes": episode,
                "new_environment_interactions": run_steps,
                "requested_total_steps": args.total_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Log: {log_path}")
    print(f"Environment interactions: {run_steps}")
    print(f"Training time: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 3600.0:.3f} hours)")
    print(f"Training summary: {summary_path}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
