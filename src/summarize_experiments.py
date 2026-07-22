from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize evaluation logs across scenarios and algorithms.")
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, default=Path("experiments") / "summary.csv")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def mean_or_nan(rows: list[dict[str, str]], key: str) -> float:
    values = numeric(rows, key)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def summarize_log(log_path: Path) -> dict[str, float | int]:
    rows = read_rows(log_path)
    return {
        "episodes": len(rows),
        "success_rate": mean_or_nan(rows, "success"),
        "collision_rate": mean_or_nan(rows, "collision"),
        "altitude_violation_rate": mean_or_nan(rows, "out_of_altitude"),
        "timeout_rate": mean_or_nan(rows, "timeout"),
        "average_reward": mean_or_nan(rows, "reward"),
        "average_steps": mean_or_nan(rows, "steps"),
        "average_final_distance": mean_or_nan(rows, "final_distance"),
        "average_path_length_m": mean_or_nan(rows, "path_length_m"),
        "average_min_depth_m": mean_or_nan(rows, "min_depth_m"),
    }


def main():
    args = parse_args()
    summary_rows = []
    for log_path in sorted(args.experiments_dir.rglob("evaluation_log.csv")):
        relative_parts = log_path.relative_to(args.experiments_dir).parts
        if len(relative_parts) not in (4, 5) or relative_parts[-2] != "results":
            continue
        scenario_name = relative_parts[0]
        algorithm_name = relative_parts[1]
        run_name = relative_parts[2] if len(relative_parts) == 5 else ""
        summary = summarize_log(log_path)
        summary_rows.append(
            {
                "scenario": scenario_name,
                "algorithm": algorithm_name,
                "run_name": run_name,
                **summary,
                "evaluation_log": str(log_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "algorithm",
        "run_name",
        "episodes",
        "success_rate",
        "collision_rate",
        "altitude_violation_rate",
        "timeout_rate",
        "average_reward",
        "average_steps",
        "average_final_distance",
        "average_path_length_m",
        "average_min_depth_m",
        "evaluation_log",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary written to: {args.output}")
    if not summary_rows:
        print("No evaluation logs found yet.")


if __name__ == "__main__":
    main()
