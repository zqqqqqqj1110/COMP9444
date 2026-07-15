from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class DroneEnvConfig:
    image_height: int = 84
    image_width: int = 84
    max_depth_m: float = 40.0
    max_steps: int = 300
    step_duration: float = 0.35
    speed_mps: float = 2.0
    vertical_speed_mps: float = 1.0
    target_position: tuple[float, float, float] = (20.0, 0.0, -3.0)
    start_position: tuple[float, float, float] = (0.0, 0.0, -3.0)
    goal_radius_m: float = 2.0
    step_penalty: float = -0.05
    distance_reward_scale: float = 2.0
    goal_reward: float = 100.0
    collision_penalty: float = -100.0
    altitude_min_z: float = -10.0
    altitude_max_z: float = -1.0
    altitude_penalty: float = -100.0
    camera_name: str = "0"
    vehicle_name: str = ""
    seed_sleep_s: float = 0.2
    spawn_settle_s: float = 0.3
    start_tolerance_m: float = 0.75


class AirSimDroneEnv(gym.Env):
    """Gymnasium-compatible AirSim multirotor navigation environment.

    Observation:
        image: depth image with shape (1, H, W), values in [0, 1]
        state: relative target vector and velocity with shape (6,)

    Actions:
        0 forward, 1 left, 2 right, 3 up, 4 down, 5 hover
    """

    metadata = {"render_modes": []}

    def __init__(self, config: DroneEnvConfig | None = None):
        super().__init__()
        self.config = config or DroneEnvConfig()
        self.airsim = self._import_airsim()
        self.client = self.airsim.MultirotorClient()
        self.client.confirmConnection()

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.config.image_height, self.config.image_width),
                    dtype=np.float32,
                ),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

        self.steps = 0
        self.previous_distance = 0.0
        self.last_info: dict[str, Any] = {}
        self._collision_baseline_time_stamp = 0.0

    @staticmethod
    def _import_airsim():
        try:
            import airsim  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'airsim' package is not installed. Install project dependencies "
                "with: pip install -r requirements.txt"
            ) from exc
        return airsim

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options

        self.client.reset()
        time.sleep(self.config.seed_sleep_s)
        self.client.enableApiControl(True, vehicle_name=self.config.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.config.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.config.vehicle_name).join()
        start_x, start_y, start_z = self.config.start_position
        start_pose = self.airsim.Pose(
            self.airsim.Vector3r(start_x, start_y, start_z),
            self.airsim.to_quaternion(0.0, 0.0, 0.0),
        )
        self.client.simSetVehiclePose(
            start_pose,
            ignore_collision=True,
            vehicle_name=self.config.vehicle_name,
        )
        self.client.moveByVelocityAsync(
            0.0,
            0.0,
            0.0,
            duration=0.2,
            vehicle_name=self.config.vehicle_name,
        ).join()
        self.client.hoverAsync(vehicle_name=self.config.vehicle_name).join()
        time.sleep(self.config.spawn_settle_s)

        position = self._position()
        start_error = math.sqrt(
            (position.x_val - start_x) ** 2
            + (position.y_val - start_y) ** 2
            + (position.z_val - start_z) ** 2
        )
        if start_error > self.config.start_tolerance_m:
            raise RuntimeError(
                "AirSim could not place the drone at the configured start position. "
                f"requested={self.config.start_position}, "
                f"actual=({position.x_val:.3f}, {position.y_val:.3f}, {position.z_val:.3f}), "
                f"error={start_error:.3f} m"
            )

        reset_collision = self.client.simGetCollisionInfo(vehicle_name=self.config.vehicle_name)
        self._collision_baseline_time_stamp = float(reset_collision.time_stamp)

        self.steps = 0
        self.previous_distance = self._distance_to_target()
        obs = self._get_observation()
        info = self._get_info()
        info.update(
            {
                "reset_collision_ignored": bool(reset_collision.has_collided),
                "reset_collision_object": reset_collision.object_name,
                "start_error_m": start_error,
            }
        )
        return obs, info

    def step(self, action: int):
        self.steps += 1
        self._apply_action(int(action))

        obs = self._get_observation()
        collision = self.client.simGetCollisionInfo(vehicle_name=self.config.vehicle_name)
        new_collision = bool(
            collision.has_collided
            and float(collision.time_stamp) > self._collision_baseline_time_stamp
        )
        distance = self._distance_to_target()
        position = self._position()

        progress = self.previous_distance - distance
        reward = self.config.step_penalty + self.config.distance_reward_scale * progress
        self.previous_distance = distance

        reached_goal = distance <= self.config.goal_radius_m
        out_of_altitude = not (self.config.altitude_min_z <= position.z_val <= self.config.altitude_max_z)
        terminated = bool(new_collision or reached_goal or out_of_altitude)
        truncated = self.steps >= self.config.max_steps

        if reached_goal:
            reward += self.config.goal_reward
        if new_collision:
            reward += self.config.collision_penalty
        if out_of_altitude:
            reward += self.config.altitude_penalty

        info = self._get_info()
        info.update(
            {
                "success": reached_goal,
                "collision": new_collision,
                "collision_object": collision.object_name if new_collision else "",
                "out_of_altitude": out_of_altitude,
                "distance_to_target": distance,
                "steps": self.steps,
            }
        )
        self.last_info = info
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.client.hoverAsync(vehicle_name=self.config.vehicle_name).join()
            self.client.armDisarm(False, vehicle_name=self.config.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.config.vehicle_name)
        except Exception:
            pass

    def _apply_action(self, action: int):
        cfg = self.config
        vx, vy, vz = 0.0, 0.0, 0.0

        if action == 0:
            vx = cfg.speed_mps
        elif action == 1:
            vy = -cfg.speed_mps
        elif action == 2:
            vy = cfg.speed_mps
        elif action == 3:
            vz = -cfg.vertical_speed_mps
        elif action == 4:
            vz = cfg.vertical_speed_mps
        elif action == 5:
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"Unknown action: {action}")

        self.client.moveByVelocityBodyFrameAsync(
            vx,
            vy,
            vz,
            duration=cfg.step_duration,
            vehicle_name=cfg.vehicle_name,
        ).join()

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {"image": self._depth_image(), "state": self._state_vector()}

    def _depth_image(self) -> np.ndarray:
        responses = self.client.simGetImages(
            [
                self.airsim.ImageRequest(
                    self.config.camera_name,
                    self.airsim.ImageType.DepthPerspective,
                    pixels_as_float=True,
                    compress=False,
                )
            ],
            vehicle_name=self.config.vehicle_name,
        )
        if not responses or responses[0].height == 0 or responses[0].width == 0:
            return np.zeros((1, self.config.image_height, self.config.image_width), dtype=np.float32)

        response = responses[0]
        depth = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
        depth = np.nan_to_num(depth, nan=self.config.max_depth_m, posinf=self.config.max_depth_m, neginf=0.0)
        depth = np.clip(depth, 0.0, self.config.max_depth_m)
        obstacle_intensity = 1.0 - (depth / self.config.max_depth_m)
        resized = cv2.resize(
            obstacle_intensity,
            (self.config.image_width, self.config.image_height),
            interpolation=cv2.INTER_AREA,
        )
        return resized.astype(np.float32)[None, :, :]

    def _state_vector(self) -> np.ndarray:
        pos = self._position()
        vel = self._velocity()
        target = self.config.target_position
        rel = np.array(
            [
                (target[0] - pos.x_val) / 50.0,
                (target[1] - pos.y_val) / 50.0,
                (target[2] - pos.z_val) / 10.0,
            ],
            dtype=np.float32,
        )
        velocity = np.array(
            [vel.x_val / 10.0, vel.y_val / 10.0, vel.z_val / 10.0],
            dtype=np.float32,
        )
        return np.concatenate([rel, velocity]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        pos = self._position()
        vel = self._velocity()
        return {
            "position": (pos.x_val, pos.y_val, pos.z_val),
            "velocity": (vel.x_val, vel.y_val, vel.z_val),
            "distance_to_target": self._distance_to_target(),
        }

    def _position(self):
        state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        return state.kinematics_estimated.position

    def _velocity(self):
        state = self.client.getMultirotorState(vehicle_name=self.config.vehicle_name)
        return state.kinematics_estimated.linear_velocity

    def _distance_to_target(self) -> float:
        pos = self._position()
        tx, ty, tz = self.config.target_position
        return math.sqrt((tx - pos.x_val) ** 2 + (ty - pos.y_val) ** 2 + (tz - pos.z_val) ** 2)
