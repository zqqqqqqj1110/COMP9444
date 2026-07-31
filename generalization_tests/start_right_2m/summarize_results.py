from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_ORDER = ("dqn_scratch", "ppo_scratch", "ppo_curriculum")


def main() -> None:
    route = json.loads((ROOT / "route.json").read_text(encoding="utf-8"))
    rows = []
    for model_name in MODEL_ORDER:
        metric_path = ROOT / "results" / model_name / "evaluation_summary.json"
        if not metric_path.is_file():
            continue
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        mode = metric["modes"][0]
        rows.append(
            {
                "route": route["name"],
                "model": model_name,
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
        )

    output_dir = ROOT / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed model metrics were found.")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"route": route, "models": rows}, indent=2), encoding="utf-8"
    )
    print(f"Summary written to: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
