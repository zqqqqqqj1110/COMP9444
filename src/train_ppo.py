from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from experiment_paths import default_model_path, ensure_experiment_dirs, print_experiment_paths, resolve_experiment_paths, write_metadata
from plot_results import plot_training_curves
from ppo_agent import PPOAgent, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO drone navigation agent in AirSim.")
    parser.add_argument("--scenario", type=str, default="blocks", help="Scenario name used for experiment outputs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run folder below the PPO experiment.")
    parser.add_argument("--resume-model", type=Path, default=None, help="PPO checkpoint used to continue training.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--target-x", type=float, default=20.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=-3.0)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for result files.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Optional override for model checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N PPO updates.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"image": obs["image"].copy(), "state": obs["state"].copy()}


def main():
    args = parse_args()
    set_seed(args.seed)

    resume_model = args.resume_model.resolve() if args.resume_model is not None else None
    if resume_model is not None and not resume_model.is_file():
        raise FileNotFoundError(f"Resume model not found: {resume_model}")

    paths = resolve_experiment_paths(
        scenario=args.scenario,
        algorithm="ppo",
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
            "script": "train_ppo.py",
            "target_position": [args.target_x, args.target_y, args.target_z],
            "start_position": [args.start_x, args.start_y, args.start_z],
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "resume_model": str(resume_model) if resume_model is not None else None,
            "resume_optimizer": resume_model is not None,
        },
    )

    env_config = DroneEnvConfig(
        max_steps=args.max_steps,
        target_position=(args.target_x, args.target_y, args.target_z),
        start_position=(args.start_x, args.start_y, args.start_z),
    )
    agent_config = PPOConfig(
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
    )

    agent = PPOAgent(agent_config)
    if resume_model is not None:
        agent.load(resume_model, load_optimizer=True)
        for parameter_group in agent.optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        print(f"Resumed PPO model and optimizer: {resume_model}")
        print(f"Resumed agent steps: {agent.steps_done}")

    env = AirSimDroneEnv(env_config)

    training_log_path = paths.results_dir / "training_log.csv"
    update_log_path = paths.results_dir / "ppo_update_log.csv"
    episode_fields = [
        "episode",
        "update",
        "global_step",
        "reward",
        "steps",
        "success",
        "collision",
        "out_of_altitude",
        "final_distance",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
    ]
    update_fields = ["update", "global_step", "rollout_size", "policy_loss", "value_loss", "entropy", "approx_kl"]

    obs, _ = env.reset()
    episode = 0
    update = 0
    global_step = agent.steps_done
    episode_reward = 0.0
    episode_steps = 0

    progress = tqdm(total=args.episodes, desc="PPO episodes")
    try:
        with training_log_path.open("w", newline="", encoding="utf-8") as train_file, update_log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as update_file:
            episode_writer = csv.DictWriter(train_file, fieldnames=episode_fields)
            update_writer = csv.DictWriter(update_file, fieldnames=update_fields)
            episode_writer.writeheader()
            update_writer.writeheader()

            while episode < args.episodes:
                rollout: list[dict] = []
                pending_episode_rows: list[dict] = []

                for _ in range(args.rollout_steps):
                    action_data = agent.act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action_data["action"])
                    done = terminated or truncated
                    rollout.append(
                        {
                            "obs": copy_observation(obs),
                            "action": action_data["action"],
                            "log_prob": action_data["log_prob"],
                            "value": action_data["value"],
                            "reward": float(reward),
                            "done": bool(done),
                        }
                    )

                    obs = next_obs
                    episode_reward += reward
                    episode_steps += 1
                    global_step += 1

                    if done:
                        episode += 1
                        pending_episode_rows.append(
                            {
                                "episode": episode,
                                "update": update + 1,
                                "global_step": global_step,
                                "reward": episode_reward,
                                "steps": info.get("steps", episode_steps),
                                "success": int(info.get("success", False)),
                                "collision": int(info.get("collision", False)),
                                "out_of_altitude": int(info.get("out_of_altitude", False)),
                                "final_distance": info.get("distance_to_target", np.nan),
                            }
                        )
                        progress.update(1)
                        if episode >= args.episodes:
                            break
                        obs, _ = env.reset()
                        episode_reward = 0.0
                        episode_steps = 0

                if not rollout:
                    break

                next_value = 0.0 if rollout[-1]["done"] else agent.value(obs)
                stats = agent.update(rollout, next_value)
                update += 1
                update_row = {
                    "update": update,
                    "global_step": global_step,
                    "rollout_size": len(rollout),
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy": stats["entropy"],
                    "approx_kl": stats["approx_kl"],
                }
                update_writer.writerow(update_row)
                update_file.flush()

                for row in pending_episode_rows:
                    row.update(
                        {
                            "policy_loss": stats["policy_loss"],
                            "value_loss": stats["value_loss"],
                            "entropy": stats["entropy"],
                            "approx_kl": stats["approx_kl"],
                        }
                    )
                    episode_writer.writerow(row)
                train_file.flush()

                if update % args.checkpoint_every == 0:
                    agent.save(paths.models_dir / f"ppo_update_{update:04d}.pt")
                    if training_log_path.exists():
                        plot_training_curves(training_log_path, paths.results_dir / "training_curves.png")
    finally:
        progress.close()
        env.close()

    final_path = default_model_path(paths)
    agent.save(final_path)
    if training_log_path.exists():
        plot_training_curves(training_log_path, paths.results_dir / "training_curves.png")
    print(f"PPO training complete. Log: {training_log_path}")
    print(f"PPO update log: {update_log_path}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
