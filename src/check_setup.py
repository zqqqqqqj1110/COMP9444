from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REQUIRED_PACKAGES = {
    "airsim": "AirSim Python client",
    "gymnasium": "RL environment API",
    "numpy": "numerical arrays",
    "cv2": "image resizing",
    "torch": "DQN/PPO model training",
    "matplotlib": "result plots",
}


def package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Check local setup for the AirSim RL project.")
    parser.add_argument("--connect", action="store_true", help="Also attempt an AirSim simulator connection.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Dependency check")
    missing = []
    for module, purpose in REQUIRED_PACKAGES.items():
        ok = package_available(module)
        status = "OK" if ok else "MISSING"
        print(f"  {module:<12} {status:<8} {purpose}")
        if not ok:
            missing.append(module)

    if missing:
        project_root = Path(__file__).resolve().parents[1]
        print()
        print("Install dependencies from the project root:")
        print(f"  cd {project_root}")
        print("  pip install -r requirements.txt")

    if args.connect:
        if not package_available("airsim"):
            print()
            print("Skipping AirSim connection because the airsim package is missing.")
            return
        try:
            import airsim  # type: ignore
        except ModuleNotFoundError as exc:
            print()
            print(f"AirSim import failed: {exc}")
            print("This is usually a dependency version issue. Reinstall the pinned requirements:")
            print("  pip install -r requirements.txt")
            return

        print()
        print("Attempting AirSim connection...")
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            state = client.getMultirotorState()
        except Exception as exc:
            print(f"AirSim connection failed: {exc}")
            print("Start Blocks.exe, wait for the 3D scene to finish loading, then run this check again.")
            return

        print("Connected.")
        print(f"Position: {state.kinematics_estimated.position}")
        print(f"Velocity: {state.kinematics_estimated.linear_velocity}")


if __name__ == "__main__":
    main()
