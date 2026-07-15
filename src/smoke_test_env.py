from __future__ import annotations

import argparse

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short AirSim environment smoke test.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--action", type=int, choices=range(6), default=0)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--target-x", type=float, default=8.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=-3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = AirSimDroneEnv(
        DroneEnvConfig(
            max_steps=args.steps,
            target_position=(args.target_x, args.target_y, args.target_z),
            start_position=(args.start_x, args.start_y, args.start_z),
        )
    )
    try:
        obs, info = env.reset()
        print(f"Reset OK. image={obs['image'].shape}, state={obs['state'].shape}")
        print(f"Initial position: {info['position']}")
        print(f"Initial distance: {info['distance_to_target']:.2f} m")
        print(
            f"Spawn error: {info['start_error_m']:.3f} m; "
            f"ignored stale collision: {info['reset_collision_ignored']} "
            f"({info['reset_collision_object']!r})"
        )

        total_reward = 0.0
        for step in range(1, args.steps + 1):
            obs, reward, terminated, truncated, info = env.step(args.action)
            total_reward += reward
            done = terminated or truncated
            print(
                f"step={step:02d} reward={reward:7.3f} "
                f"distance={info['distance_to_target']:6.2f}m "
                f"collision={info['collision']} done={done}"
            )
            if done:
                break

        print(f"Smoke test complete. total_reward={total_reward:.3f}")
        if args.require_clean and (info.get("collision", False) or info.get("out_of_altitude", False)):
            raise RuntimeError("Smoke test failed because the drone collided or left the altitude range.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
