from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import deque
from datetime import datetime
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
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="Also restore Adam state. Disabled by default for curriculum stage changes.",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Stop after this many new environment interactions instead of using the episode limit.",
    )
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
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=0,
        help="Also save after each N new environment interactions; 0 disables step checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.add_argument("--value-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--best-min-episodes", type=int, default=20)
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
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.total_steps is not None and args.total_steps <= 0:
        raise ValueError("--total-steps must be positive when provided.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps cannot be negative.")
    if args.reward_scale <= 0:
        raise ValueError("--reward-scale must be positive.")
    if args.best_window <= 0 or args.best_min_episodes <= 0:
        raise ValueError("--best-window and --best-min-episodes must be positive.")
    if args.best_min_episodes > args.best_window:
        raise ValueError("--best-min-episodes cannot exceed --best-window.")
    set_seed(args.seed)
    training_started_at = datetime.now()
    training_started_perf = time.perf_counter()

    resume_model = args.resume_model.resolve() if args.resume_model is not None else None
    if resume_model is not None and not resume_model.is_file():
        raise FileNotFoundError(f"Resume model not found: {resume_model}")
    if resume_model is not None and not PPOAgent.checkpoint_uses_stable_features(resume_model):
        raise ValueError(
            "The resume checkpoint predates PPO feature stabilisation and cannot be used for a new stable run. "
            "Start Stage 1 from scratch, or resume from a checkpoint produced by this version."
        )

    env_config = DroneEnvConfig(
        max_steps=args.max_steps,
        target_position=(args.target_x, args.target_y, args.target_z),
        start_position=(args.start_x, args.start_y, args.start_z),
    )

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
            "total_steps": args.total_steps,
            "stop_condition": "total_steps" if args.total_steps is not None else "episodes",
            "max_steps": args.max_steps,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "reward_scale": args.reward_scale,
            "value_loss": args.value_loss,
            "normalize_shared_features": True,
            "best_window": args.best_window,
            "best_min_episodes": args.best_min_episodes,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "seed": args.seed,
            "resume_model": str(resume_model) if resume_model is not None else None,
            "resume_optimizer": bool(resume_model is not None and args.resume_optimizer),
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

    agent_config = PPOConfig(
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        value_loss_type=args.value_loss,
        normalize_shared_features=True,
    )

    agent = PPOAgent(agent_config)
    if resume_model is not None:
        agent.load(resume_model, load_optimizer=args.resume_optimizer)
        for parameter_group in agent.optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        restored = "model and optimizer" if args.resume_optimizer else "model weights; optimizer reset"
        print(f"Resumed PPO {restored}: {resume_model}")
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
        "explained_variance",
        "activation_saturation",
        "activation_abs_mean",
        "max_action_probability",
        "timeout",
        "final_x",
        "final_y",
        "final_z",
        "path_length_m",
        "min_depth_m",
        "dominant_action",
        "dominant_action_fraction",
    ]
    update_fields = [
        "update",
        "global_step",
        "rollout_size",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "explained_variance",
        "activation_saturation",
        "activation_abs_mean",
        "max_action_probability",
    ]

    obs, _ = env.reset()
    episode = 0
    update = 0
    global_step = agent.steps_done
    episode_reward = 0.0
    episode_steps = 0
    run_steps = 0
    last_step_checkpoint = 0
    episode_action_counts = np.zeros(agent_config.action_dim, dtype=np.int64)
    recent_episodes: deque[dict] = deque(maxlen=args.best_window)
    best_score: tuple[float, float, float] | None = None
    best_metrics: dict[str, float | int] | None = None
    best_model_path = paths.models_dir / "ppo_best.pt"

    use_step_budget = args.total_steps is not None
    progress = tqdm(
        total=args.total_steps if use_step_budget else args.episodes,
        desc="PPO steps" if use_step_budget else "PPO episodes",
        unit="step" if use_step_budget else "episode",
    )
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

            while (run_steps < args.total_steps) if use_step_budget else (episode < args.episodes):
                rollout: list[dict] = []
                pending_episode_rows: list[dict] = []

                rollout_limit = args.rollout_steps
                if use_step_budget:
                    rollout_limit = min(rollout_limit, args.total_steps - run_steps)

                for _ in range(rollout_limit):
                    action_data = agent.act(obs)
                    episode_action_counts[action_data["action"]] += 1
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
                    run_steps += 1
                    if use_step_budget:
                        progress.update(1)

                    if done:
                        episode += 1
                        position = info.get("position", (np.nan, np.nan, np.nan))
                        dominant_action = int(np.argmax(episode_action_counts))
                        dominant_action_fraction = float(episode_action_counts[dominant_action] / max(episode_steps, 1))
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
                                "timeout": int(bool(truncated and not terminated)),
                                "final_x": position[0],
                                "final_y": position[1],
                                "final_z": position[2],
                                "path_length_m": info.get("path_length_m", np.nan),
                                "min_depth_m": info.get("episode_min_depth_m", np.nan),
                                "dominant_action": dominant_action,
                                "dominant_action_fraction": dominant_action_fraction,
                            }
                        )
                        if not use_step_budget:
                            progress.update(1)
                        if (not use_step_budget) and episode >= args.episodes:
                            break
                        if use_step_budget and run_steps >= args.total_steps:
                            break
                        obs, _ = env.reset()
                        episode_reward = 0.0
                        episode_steps = 0
                        episode_action_counts.fill(0)

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
                    "explained_variance": stats["explained_variance"],
                    "activation_saturation": stats["activation_saturation"],
                    "activation_abs_mean": stats["activation_abs_mean"],
                    "max_action_probability": stats["max_action_probability"],
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
                            "explained_variance": stats["explained_variance"],
                            "activation_saturation": stats["activation_saturation"],
                            "activation_abs_mean": stats["activation_abs_mean"],
                            "max_action_probability": stats["max_action_probability"],
                        }
                    )
                    episode_writer.writerow(row)
                    recent_episodes.append(row)
                train_file.flush()

                if len(recent_episodes) >= args.best_min_episodes:
                    success_rate = float(np.mean([row["success"] for row in recent_episodes]))
                    unsafe_rate = float(
                        np.mean(
                            [
                                bool(row["collision"] or row["out_of_altitude"])
                                for row in recent_episodes
                            ]
                        )
                    )
                    mean_distance = float(np.mean([row["final_distance"] for row in recent_episodes]))
                    candidate_score = (success_rate, -unsafe_rate, -mean_distance)
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_metrics = {
                            "success_rate": success_rate,
                            "unsafe_rate": unsafe_rate,
                            "mean_final_distance": mean_distance,
                            "update": update,
                            "global_step": global_step,
                        }
                        agent.save(best_model_path)
                        print(
                            "Saved PPO best model: "
                            f"success={success_rate:.1%}, unsafe={unsafe_rate:.1%}, "
                            f"distance={mean_distance:.2f} m"
                        )

                if update % args.checkpoint_every == 0:
                    agent.save(paths.models_dir / f"ppo_update_{update:04d}.pt")
                    if training_log_path.exists():
                        plot_training_curves(training_log_path, paths.results_dir / "training_curves.png")

                if (
                    args.checkpoint_every_steps > 0
                    and run_steps - last_step_checkpoint >= args.checkpoint_every_steps
                ):
                    agent.save(paths.models_dir / f"ppo_step_{run_steps:07d}.pt")
                    last_step_checkpoint = run_steps
    finally:
        progress.close()
        env.close()

    final_path = default_model_path(paths)
    agent.save(final_path)
    if not best_model_path.exists():
        agent.save(best_model_path)
    if training_log_path.exists():
        plot_training_curves(training_log_path, paths.results_dir / "training_curves.png")
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
                "agent_cumulative_steps": agent.steps_done,
                "requested_total_steps": args.total_steps,
                "resume_model": str(resume_model) if resume_model is not None else None,
                "resume_optimizer": bool(resume_model is not None and args.resume_optimizer),
                "best_model": str(best_model_path),
                "best_window_metrics": best_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PPO training complete. Log: {training_log_path}")
    print(f"PPO update log: {update_log_path}")
    print(f"New environment interactions: {run_steps}")
    print(f"Training time: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 3600.0:.3f} hours)")
    print(f"Training summary: {summary_path}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
