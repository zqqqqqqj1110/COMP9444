from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from evaluate import load_agent


ACTION_NAMES = ("forward", "left", "right", "up", "down", "hover")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained AirSim navigation policy and record annotated camera videos."
    )
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=110)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--start-z", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--stop-after-success",
        action="store_true",
        help="Stop after recording the first successful episode.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes and max-steps must be positive")
    if args.video_width < 320 or args.video_height < 240 or args.fps <= 0:
        raise ValueError("video dimensions must be at least 320x240 and fps must be positive")
    if args.algorithm == "dqn" and args.policy_mode == "stochastic":
        raise ValueError("stochastic inference is supported only for PPO")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")


def capture_scene_frame(env: AirSimDroneEnv, camera_name: str) -> np.ndarray:
    responses = env.client.simGetImages(
        [
            env.airsim.ImageRequest(
                camera_name,
                env.airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            )
        ],
        vehicle_name=env.config.vehicle_name,
    )
    if not responses or responses[0].height == 0 or responses[0].width == 0:
        raise RuntimeError(f"AirSim camera '{camera_name}' returned an empty scene image")

    response = responses[0]
    pixels = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    pixel_count = response.height * response.width
    if pixels.size == pixel_count * 3:
        rgb = pixels.reshape(response.height, response.width, 3)
    elif pixels.size == pixel_count * 4:
        rgb = pixels.reshape(response.height, response.width, 4)[:, :, :3]
    else:
        raise RuntimeError(
            f"Unexpected AirSim scene image size: {pixels.size} bytes for "
            f"{response.width}x{response.height}"
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def add_hud(
    frame: np.ndarray,
    depth_observation: np.ndarray,
    episode: int,
    step: int,
    action: int | None,
    reward: float,
    distance: float,
    position: tuple[float, float, float],
    policy_mode: str,
    status: str,
) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (12, 12), (540, 139), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0.0, output)

    action_label = "reset" if action is None else f"{action}: {ACTION_NAMES[action]}"
    lines = (
        f"Episode {episode} | Step {step} | {policy_mode}",
        f"Action {action_label} | Return {reward:+.2f}",
        f"Distance {distance:.2f} m | Status {status}",
        f"Position ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (26, 39 + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    depth = np.clip(depth_observation.squeeze(), 0.0, 1.0)
    depth = (depth * 255.0).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_TURBO)
    inset_size = min(180, output.shape[0] // 3, output.shape[1] // 4)
    depth = cv2.resize(depth, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)
    inset_x = output.shape[1] - inset_size - 14
    inset_y = 14
    output[inset_y : inset_y + inset_size, inset_x : inset_x + inset_size] = depth
    cv2.rectangle(
        output,
        (inset_x - 1, inset_y - 1),
        (inset_x + inset_size, inset_y + inset_size),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        output,
        "Policy depth input",
        (inset_x, inset_y + inset_size + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def episode_status(info: dict, terminated: bool, truncated: bool) -> str:
    if info.get("success", False):
        return "success"
    if info.get("collision", False):
        return "collision"
    if info.get("out_of_altitude", False):
        return "altitude"
    if truncated and not terminated:
        return "timeout"
    return "running"


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env = AirSimDroneEnv(
        DroneEnvConfig(
            max_steps=args.max_steps,
            target_position=(args.target_x, args.target_y, args.target_z),
            start_position=(args.start_x, args.start_y, args.start_z),
            camera_name=args.camera_name,
        )
    )
    agent = load_agent(args.algorithm, args.model)
    deterministic = args.policy_mode == "deterministic"
    rows: list[dict] = []
    episode_rows: list[dict] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, reset_info = env.reset(seed=args.seed + episode - 1)
            total_reward = 0.0
            terminated = False
            truncated = False
            info = reset_info
            action_counts = np.zeros(agent.config.action_dim, dtype=np.int64)
            temporary_path = args.output_dir / f"episode_{episode:03d}.mp4"
            writer = cv2.VideoWriter(
                str(temporary_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (args.video_width, args.video_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video: {temporary_path}")

            last_annotated_frame: np.ndarray | None = None
            try:
                scene = fit_frame(
                    capture_scene_frame(env, args.camera_name),
                    args.video_width,
                    args.video_height,
                )
                initial_position = reset_info["position"]
                last_annotated_frame = add_hud(
                    scene,
                    observation["image"],
                    episode,
                    0,
                    None,
                    0.0,
                    float(reset_info["distance_to_target"]),
                    initial_position,
                    args.policy_mode,
                    "ready",
                )
                writer.write(last_annotated_frame)

                for step in range(1, args.max_steps + 1):
                    action = agent.select_action(observation, evaluate=deterministic)
                    action_counts[action] += 1
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    status = episode_status(info, terminated, truncated)
                    position = info["position"]

                    scene = fit_frame(
                        capture_scene_frame(env, args.camera_name),
                        args.video_width,
                        args.video_height,
                    )
                    annotated_frame = add_hud(
                        scene,
                        observation["image"],
                        episode,
                        step,
                        action,
                        total_reward,
                        float(info["distance_to_target"]),
                        position,
                        args.policy_mode,
                        status,
                    )
                    last_annotated_frame = annotated_frame
                    writer.write(annotated_frame)
                    rows.append(
                        {
                            "episode": episode,
                            "step": step,
                            "action": action,
                            "action_name": ACTION_NAMES[action],
                            "reward": reward,
                            "cumulative_reward": total_reward,
                            "distance_to_target": info["distance_to_target"],
                            "x": position[0],
                            "y": position[1],
                            "z": position[2],
                            "success": int(info.get("success", False)),
                            "collision": int(info.get("collision", False)),
                            "out_of_altitude": int(info.get("out_of_altitude", False)),
                            "timeout": int(bool(truncated and not terminated)),
                        }
                    )
                    if terminated or truncated:
                        for _ in range(max(1, int(round(args.fps * 1.5)))):
                            writer.write(annotated_frame)
                        break
            finally:
                writer.release()

            status = episode_status(info, terminated, truncated)
            final_path = args.output_dir / f"episode_{episode:03d}_{status}.mp4"
            temporary_path.replace(final_path)
            preview_path = args.output_dir / f"episode_{episode:03d}_{status}_preview.jpg"
            if last_annotated_frame is None or not cv2.imwrite(str(preview_path), last_annotated_frame):
                raise RuntimeError(f"Could not create video preview: {preview_path}")
            observed_steps = max(int(action_counts.sum()), 1)
            dominant_action = int(action_counts.argmax())
            episode_rows.append(
                {
                    "episode": episode,
                    "status": status,
                    "success": int(info.get("success", False)),
                    "collision": int(info.get("collision", False)),
                    "out_of_altitude": int(info.get("out_of_altitude", False)),
                    "timeout": int(bool(truncated and not terminated)),
                    "reward": total_reward,
                    "steps": int(info.get("steps", observed_steps)),
                    "final_distance": info.get("distance_to_target", np.nan),
                    "path_length_m": info.get("path_length_m", np.nan),
                    "min_depth_m": info.get("episode_min_depth_m", np.nan),
                    "dominant_action": dominant_action,
                    "dominant_action_name": ACTION_NAMES[dominant_action],
                    "dominant_action_fraction": float(action_counts[dominant_action] / observed_steps),
                    "video": str(final_path.resolve()),
                    "preview": str(preview_path.resolve()),
                }
            )
            print(
                f"Episode {episode}: {status}, reward={total_reward:.2f}, "
                f"steps={observed_steps}, distance={float(info.get('distance_to_target', np.nan)):.2f} m"
            )
            print(f"  Video: {final_path}")
            if args.stop_after_success and info.get("success", False):
                break
    finally:
        env.close()

    step_log_path = args.output_dir / "inference_steps.csv"
    with step_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    episode_log_path = args.output_dir / "inference_episodes.csv"
    with episode_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    successes = sum(row["success"] for row in episode_rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": args.algorithm,
        "policy_mode": args.policy_mode,
        "model": str(args.model.resolve()),
        "seed": args.seed,
        "start": [args.start_x, args.start_y, args.start_z],
        "target": [args.target_x, args.target_y, args.target_z],
        "attempted_episodes": len(episode_rows),
        "successful_episodes": successes,
        "success_rate": successes / len(episode_rows),
        "step_log": str(step_log_path.resolve()),
        "episode_log": str(episode_log_path.resolve()),
    }
    summary_path = args.output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print(f"Inference complete: {len(episode_rows)} episode(s), success rate={summary['success_rate']:.1%}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
