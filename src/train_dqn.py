from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from dqn_agent import DQNAgent, DQNConfig
from experiment_paths import default_model_path, ensure_experiment_dirs, print_experiment_paths, resolve_experiment_paths, write_metadata
from plot_results import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN drone navigation agent in AirSim.")
    parser.add_argument("--scenario", type=str, default="blocks", help="Scenario name used for experiment outputs.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-x", type=float, default=20.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for result files.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Optional override for model checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=25)
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
    set_seed(args.seed)
    paths = resolve_experiment_paths(
        scenario=args.scenario,
        algorithm="dqn",
        output_root=args.output_root,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
    )
    ensure_experiment_dirs(paths)
    print_experiment_paths(paths)
    write_metadata(
        paths,
        {
            "script": "train_dqn.py",
            "target_position": [args.target_x, args.target_y, args.target_z],
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    env_config = DroneEnvConfig(
        max_steps=args.max_steps,
        target_position=(args.target_x, args.target_y, args.target_z),
    )
    agent_config = DQNConfig()

    env = AirSimDroneEnv(env_config)
    agent = DQNAgent(agent_config)

    log_path = paths.results_dir / "training_log.csv"
    fieldnames = [
        "episode",
        "reward",
        "steps",
        "success",
        "collision",
        "out_of_altitude",
        "final_distance",
        "epsilon",
        "mean_loss",
    ]

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            for episode in trange(1, args.episodes + 1, desc="Training"):
                obs, _ = env.reset()
                total_reward = 0.0
                losses: list[float] = []
                info = {}

                for _ in range(args.max_steps):
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    agent.remember(obs, action, reward, next_obs, done)
                    loss = agent.optimize()
                    if loss is not None:
                        losses.append(loss)

                    obs = next_obs
                    total_reward += reward
                    if done:
                        break

                row = {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": info.get("steps", args.max_steps),
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "out_of_altitude": int(info.get("out_of_altitude", False)),
                    "final_distance": info.get("distance_to_target", np.nan),
                    "epsilon": agent.epsilon(),
                    "mean_loss": float(np.mean(losses)) if losses else np.nan,
                }
                writer.writerow(row)
                f.flush()

                if episode % args.checkpoint_every == 0:
                    agent.save(paths.models_dir / f"dqn_episode_{episode:04d}.pt")
                    plot_training_curves(log_path, paths.results_dir / "training_curves.png")

        finally:
            env.close()

    final_path = default_model_path(paths)
    agent.save(final_path)
    plot_training_curves(log_path, paths.results_dir / "training_curves.png")
    print(f"Training complete. Log: {log_path}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
