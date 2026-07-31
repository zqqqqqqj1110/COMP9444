from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    route = json.loads((ROOT / "route.json").read_text(encoding="utf-8"))
    result_path = ROOT / "results" / "ppo_scratch" / "evaluation_summary.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Formal result not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mode = result["modes"][0]
    row = {
        "route": route["name"],
        "model": "ppo_scratch",
        "policy_mode": mode["policy_mode"],
        "episodes": mode["episodes"],
        "success_rate": mode["success_rate"],
        "collision_rate": mode["collision_rate"],
        "altitude_violation_rate": mode["altitude_violation_rate"],
        "unsafe_rate": mode["unsafe_rate"],
        "timeout_rate": mode["timeout_rate"],
        "average_reward": mode["average_reward"],
        "average_steps": mode["average_steps"],
        "average_final_distance": mode["average_final_distance"],
        "average_path_length_m": mode["average_path_length_m"],
        "average_min_depth_m": mode["average_min_depth_m"],
    }
    output_dir = ROOT / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (output_dir / "summary.json").write_text(
        json.dumps({"route": route, "model": row}, indent=2), encoding="utf-8"
    )
    print(f"Summary written to: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
