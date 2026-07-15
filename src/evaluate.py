from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from dqn_agent import DQNAgent, DQNConfig
from experiment_paths import default_model_path, ensure_experiment_dirs, print_experiment_paths, resolve_experiment_paths
from ppo_agent import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained drone navigation agent in AirSim.")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--scenario", type=str, default="blocks", help="Scenario name used for experiment outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder used during training.")
    parser.add_argument("--model", type=Path, default=None)
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
    return parser.parse_args()


def load_agent(algorithm: str, model_path: Path):
    if algorithm == "dqn":
        agent = DQNAgent(DQNConfig())
    elif algorithm == "ppo":
        agent = PPOAgent(PPOConfig())
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    agent.load(model_path)
    return agent


def main():
    args = parse_args()
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

    rows = []
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            info = {}

            for _ in range(args.max_steps):
                action = agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            rows.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": info.get("steps", args.max_steps),
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "out_of_altitude": int(info.get("out_of_altitude", False)),
                    "final_distance": info.get("distance_to_target", np.nan),
                }
            )
    finally:
        env.close()

    log_path = paths.results_dir / "evaluation_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    success_rate = np.mean([row["success"] for row in rows])
    collision_rate = np.mean([row["collision"] for row in rows])
    avg_reward = np.mean([row["reward"] for row in rows])
    avg_steps = np.mean([row["steps"] for row in rows])

    print(f"Evaluation log: {log_path}")
    print(f"Model: {model_path}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Collision rate: {collision_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average steps: {avg_steps:.1f}")


if __name__ == "__main__":
    main()
