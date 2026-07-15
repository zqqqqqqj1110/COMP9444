from __future__ import annotations

import argparse
from pathlib import Path

from experiment_paths import slugify


def parse_args():
    parser = argparse.ArgumentParser(description="List AirSim scene executables and suggested scenario names.")
    parser.add_argument("--airsim-root", type=Path, default=Path(r"D:\AirSim\Blocks"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.airsim_root.exists():
        print(f"AirSim scene root not found: {args.airsim_root}")
        return

    scene_exes = []
    for exe_path in sorted(args.airsim_root.rglob("*.exe")):
        parts = {part.lower() for part in exe_path.parts}
        if "engine" in parts or "binaries" in parts:
            continue
        scene_exes.append(exe_path)

    if not scene_exes:
        print(f"No scene executables found under: {args.airsim_root}")
        return

    print("Found AirSim scene executables:")
    for exe_path in scene_exes:
        print(f"  scenario={slugify(exe_path.stem):<24} exe={exe_path}")


if __name__ == "__main__":
    main()
