from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from airsim_drone_env import AirSimDroneEnv, DroneEnvConfig
from evaluate import (
    evaluate_policy,
    evaluation_episode_fields,
    load_agent,
    set_seed,
    summarize_rows,
    write_csv,
)
from experiment_paths import ensure_experiment_dirs, print_experiment_paths, resolve_experiment_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate DQN or PPO checkpoints in two stages and "
            "select one deployment model."
        )
    )
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Compatibility alias for --stage1-episodes.",
    )
    parser.add_argument("--stage1-episodes", type=int, default=None)
    parser.add_argument("--stage2-episodes", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stage2-seed-offset", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--start-z", type=float, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def discover_checkpoints(models_dir: Path, algorithm: str) -> list[Path]:
    checkpoints = sorted(models_dir.glob(f"{algorithm}_step_*.pt"))
    extra_names = [f"{algorithm}_final.pt"]
    if algorithm == "ppo":
        extra_names.insert(0, "ppo_best.pt")
    for name in extra_names:
        path = models_dir / name
        if path.is_file():
            checkpoints.append(path)
    return list(dict.fromkeys(checkpoints))


def checkpoint_run_step(path: Path, algorithm: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(algorithm)}_step_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def sweep_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["success_rate"]),
        -float(summary["unsafe_rate"]),
        -float(summary["average_final_distance"]),
        -float(summary["average_steps"]),
    )


def rank_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=sweep_score, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def evaluate_checkpoints(
    env: AirSimDroneEnv,
    algorithm: str,
    checkpoints: list[Path],
    episodes: int,
    max_steps: int,
    seed: int,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for index, checkpoint in enumerate(checkpoints, start=1):
        set_seed(seed)
        agent = load_agent(algorithm, checkpoint)
        rows, _ = evaluate_policy(
            env=env,
            agent=agent,
            algorithm=algorithm,
            policy_mode="deterministic",
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            collect_trajectory=False,
        )
        summary = summarize_rows(rows)
        run_step = checkpoint_run_step(checkpoint, algorithm)
        summary_rows.append(
            {
                "phase": phase,
                "checkpoint": checkpoint.name,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_run_step": run_step,
                "agent_cumulative_steps": agent.steps_done,
                **summary,
            }
        )
        for row in rows:
            episode_rows.append(
                {
                    "phase": phase,
                    "checkpoint": checkpoint.name,
                    "checkpoint_run_step": run_step,
                    "agent_cumulative_steps": agent.steps_done,
                    **row,
                }
            )
        print(
            f"[{index}/{len(checkpoints)}] {checkpoint.name}: "
            f"success={float(summary['success_rate']):.1%}, "
            f"unsafe={float(summary['unsafe_rate']):.1%}, "
            f"distance={float(summary['average_final_distance']):.2f} m"
        )

    return rank_summaries(summary_rows), episode_rows


def write_phase_outputs(
    results_dir: Path,
    phase: str,
    summary_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
):
    write_csv(
        results_dir / f"checkpoint_sweep_{phase}.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    write_csv(
        results_dir / f"checkpoint_sweep_{phase}_episodes.csv",
        [
            "phase",
            "checkpoint",
            "checkpoint_run_step",
            "agent_cumulative_steps",
            *evaluation_episode_fields(),
        ],
        episode_rows,
    )


def main():
    args = parse_args()
    stage1_episodes = (
        args.stage1_episodes if args.stage1_episodes is not None else args.episodes
    )
    if (
        stage1_episodes <= 0
        or args.stage2_episodes <= 0
        or args.top_k <= 0
        or args.max_steps <= 0
    ):
        raise ValueError(
            "Stage episode counts, --top-k, and --max-steps must be positive."
        )
    if args.stage2_seed_offset == 0:
        raise ValueError("--stage2-seed-offset must be non-zero.")

    paths = resolve_experiment_paths(
        scenario=args.scenario,
        algorithm=args.algorithm,
        output_root=args.output_root,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        run_name=args.run_name,
    )
    ensure_experiment_dirs(paths)
    print_experiment_paths(paths)
    checkpoints = discover_checkpoints(paths.models_dir, args.algorithm)
    if not checkpoints:
        raise FileNotFoundError(
            f"No {args.algorithm.upper()} checkpoints found in: {paths.models_dir}"
        )

    env = AirSimDroneEnv(
        DroneEnvConfig(
            max_steps=args.max_steps,
            target_position=(args.target_x, args.target_y, args.target_z),
            start_position=(args.start_x, args.start_y, args.start_z),
        )
    )
    try:
        print("")
        print(
            f"Stage 1: {len(checkpoints)} checkpoints x "
            f"{stage1_episodes} episodes"
        )
        stage1_rows, stage1_episode_rows = evaluate_checkpoints(
            env=env,
            algorithm=args.algorithm,
            checkpoints=checkpoints,
            episodes=stage1_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            phase="stage1",
        )
        finalists = stage1_rows[: min(args.top_k, len(stage1_rows))]
        finalist_paths = [Path(row["checkpoint_path"]) for row in finalists]
        finalist_names = {path.name for path in finalist_paths}
        for row in stage1_rows:
            row["advanced"] = int(row["checkpoint"] in finalist_names)
            row["selected"] = 0
        write_phase_outputs(
            paths.results_dir,
            "stage1",
            stage1_rows,
            stage1_episode_rows,
        )

        stage2_seed = args.seed + args.stage2_seed_offset
        print("")
        print(
            f"Stage 2: {len(finalist_paths)} finalists x "
            f"{args.stage2_episodes} episodes (seed={stage2_seed})"
        )
        stage2_rows, stage2_episode_rows = evaluate_checkpoints(
            env=env,
            algorithm=args.algorithm,
            checkpoints=finalist_paths,
            episodes=args.stage2_episodes,
            max_steps=args.max_steps,
            seed=stage2_seed,
            phase="stage2",
        )
    finally:
        env.close()

    best_path = Path(stage2_rows[0]["checkpoint_path"])
    for row in stage2_rows:
        row["advanced"] = 1
        row["selected"] = int(
            Path(row["checkpoint_path"]).resolve() == best_path.resolve()
        )

    selected_model_path = (
        paths.models_dir / f"{args.algorithm}_best_deterministic.pt"
    )
    previous_selection_backup: Path | None = None
    if selected_model_path.is_file():
        previous_selection_backup = (
            paths.models_dir
            / f"{args.algorithm}_best_deterministic_single_stage.pt"
        )
        if not previous_selection_backup.exists():
            shutil.copy2(selected_model_path, previous_selection_backup)
    if best_path.resolve() != selected_model_path.resolve():
        shutil.copy2(best_path, selected_model_path)

    write_phase_outputs(
        paths.results_dir,
        "stage2",
        stage2_rows,
        stage2_episode_rows,
    )
    combined_rows = [*stage1_rows, *stage2_rows]
    write_csv(
        paths.results_dir / "checkpoint_sweep_two_stage.csv",
        list(combined_rows[0].keys()),
        combined_rows,
    )
    combined_episode_rows = [*stage1_episode_rows, *stage2_episode_rows]
    write_csv(
        paths.results_dir / "checkpoint_sweep_two_stage_episodes.csv",
        [
            "phase",
            "checkpoint",
            "checkpoint_run_step",
            "agent_cumulative_steps",
            *evaluation_episode_fields(),
        ],
        combined_episode_rows,
    )

    summary_path = paths.results_dir / "checkpoint_sweep_two_stage_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "algorithm": args.algorithm,
                "policy_mode": "deterministic",
                "selection_protocol": "two_stage",
                "stage1": {
                    "episodes_per_checkpoint": stage1_episodes,
                    "seed": args.seed,
                    "candidate_count": len(checkpoints),
                    "advanced_count": len(finalist_paths),
                    "advanced_checkpoints": [path.name for path in finalist_paths],
                },
                "stage2": {
                    "episodes_per_checkpoint": args.stage2_episodes,
                    "seed": stage2_seed,
                    "candidate_count": len(finalist_paths),
                },
                "selected_source_checkpoint": str(best_path.resolve()),
                "selected_model": str(selected_model_path.resolve()),
                "previous_single_stage_model_backup": (
                    str(previous_selection_backup.resolve())
                    if previous_selection_backup is not None
                    else None
                ),
                "selection_order": [
                    "success_rate descending",
                    "unsafe_rate ascending",
                    "average_final_distance ascending",
                    "average_steps ascending",
                ],
                "final_test_note": (
                    "Run a separate evaluation with a new seed after selection; "
                    "do not report Stage 1 or Stage 2 selection episodes as test results."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"Stage 1 results: {paths.results_dir / 'checkpoint_sweep_stage1.csv'}")
    print(f"Stage 2 results: {paths.results_dir / 'checkpoint_sweep_stage2.csv'}")
    print(f"Selected source: {best_path}")
    print(f"Deterministic best model: {selected_model_path}")
    if previous_selection_backup is not None:
        print(f"Previous single-stage model: {previous_selection_backup}")
    print("Run a fresh-seed final test before reporting this model.")


if __name__ == "__main__":
    main()
