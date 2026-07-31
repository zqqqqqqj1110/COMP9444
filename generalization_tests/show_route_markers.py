from __future__ import annotations

import argparse
import json
from pathlib import Path

import airsim


def parse_args():
    parser = argparse.ArgumentParser(description="Show persistent route markers in AirSim.")
    parser.add_argument("route", type=Path, help="Route JSON file to display.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing persistent markers instead of clearing them first.",
    )
    parser.add_argument(
        "--focus-start",
        action="store_true",
        help="Move the drone above START so the route markers are easy to find.",
    )
    return parser.parse_args()


def vector(point: dict[str, float]) -> airsim.Vector3r:
    return airsim.Vector3r(float(point["x"]), float(point["y"]), float(point["z"]))


def main():
    args = parse_args()
    route_path = args.route.resolve()
    route = json.loads(route_path.read_text(encoding="utf-8"))
    start = vector(route["start"])
    target = vector(route["target"])

    client = airsim.MultirotorClient()
    client.confirmConnection()
    if not args.keep_existing:
        client.simFlushPersistentMarkers()

    client.simPlotLineStrip(
        [start, target],
        color_rgba=[1.0, 1.0, 0.0, 1.0],
        thickness=8.0,
        duration=-1.0,
        is_persistent=True,
    )
    client.simPlotPoints(
        [start],
        color_rgba=[0.0, 1.0, 0.0, 1.0],
        size=35.0,
        duration=-1.0,
        is_persistent=True,
    )
    client.simPlotPoints(
        [target],
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        size=35.0,
        duration=-1.0,
        is_persistent=True,
    )
    client.simPlotLineStrip(
        [start, airsim.Vector3r(start.x_val, start.y_val, start.z_val - 15.0)],
        color_rgba=[0.0, 1.0, 0.0, 1.0],
        thickness=14.0,
        duration=-1.0,
        is_persistent=True,
    )
    client.simPlotLineStrip(
        [target, airsim.Vector3r(target.x_val, target.y_val, target.z_val - 15.0)],
        color_rgba=[1.0, 0.0, 0.0, 1.0],
        thickness=14.0,
        duration=-1.0,
        is_persistent=True,
    )
    client.simPlotStrings(
        ["START", "TARGET"],
        [
            airsim.Vector3r(start.x_val, start.y_val, start.z_val - 1.5),
            airsim.Vector3r(target.x_val, target.y_val, target.z_val - 1.5),
        ],
        scale=2.0,
        color_rgba=[1.0, 1.0, 1.0, 1.0],
        duration=-1.0,
    )

    if args.focus_start:
        client.enableApiControl(True)
        client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(start.x_val, start.y_val, start.z_val - 4.0),
                airsim.to_quaternion(0.0, 0.0, 0.0),
            ),
            ignore_collision=True,
        )
        client.enableApiControl(False)

    print(f"Route: {route.get('name', route_path.stem)}")
    print(f"START: ({start.x_val:.3f}, {start.y_val:.3f}, {start.z_val:.3f})")
    print(f"TARGET: ({target.x_val:.3f}, {target.y_val:.3f}, {target.z_val:.3f})")
    print("Markers displayed: START=green, TARGET=red, route line=yellow")
    if args.focus_start:
        print("Drone moved 4 m above START for viewing")


if __name__ == "__main__":
    main()
