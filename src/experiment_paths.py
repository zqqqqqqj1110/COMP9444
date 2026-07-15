from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_ALGORITHMS = {"dqn", "ppo"}


@dataclass(frozen=True)
class ExperimentPaths:
    scenario: str
    algorithm: str
    run_name: str | None
    root_dir: Path
    results_dir: Path
    models_dir: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    slug = slug.strip("._-")
    if not slug:
        raise ValueError("Scenario or algorithm name cannot be empty.")
    return slug


def resolve_experiment_paths(
    scenario: str,
    algorithm: str,
    output_root: str | Path = "experiments",
    results_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
    run_name: str | None = None,
) -> ExperimentPaths:
    scenario_slug = slugify(scenario)
    algorithm_slug = slugify(algorithm)
    if algorithm_slug not in SUPPORTED_ALGORITHMS:
        supported = ", ".join(sorted(SUPPORTED_ALGORITHMS))
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {supported}")

    run_slug = slugify(run_name) if run_name else None
    root_dir = Path(output_root) / scenario_slug / algorithm_slug
    if run_slug is not None:
        root_dir = root_dir / run_slug
    resolved_results = Path(results_dir) if results_dir is not None else root_dir / "results"
    resolved_models = Path(models_dir) if models_dir is not None else root_dir / "models"
    return ExperimentPaths(
        scenario=scenario_slug,
        algorithm=algorithm_slug,
        run_name=run_slug,
        root_dir=root_dir,
        results_dir=resolved_results,
        models_dir=resolved_models,
    )


def ensure_experiment_dirs(paths: ExperimentPaths):
    paths.root_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)


def final_model_name(algorithm: str) -> str:
    algorithm_slug = slugify(algorithm)
    return f"{algorithm_slug}_final.pt"


def default_model_path(paths: ExperimentPaths) -> Path:
    return paths.models_dir / final_model_name(paths.algorithm)


def write_metadata(paths: ExperimentPaths, extra: dict[str, Any]):
    metadata = {
        "scenario": paths.scenario,
        "algorithm": paths.algorithm,
        "run_name": paths.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            key: str(value)
            for key, value in asdict(paths).items()
            if isinstance(value, Path)
        },
    }
    metadata.update(extra)
    metadata_path = paths.root_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_experiment_paths(paths: ExperimentPaths):
    print(f"Scenario: {paths.scenario}")
    print(f"Algorithm: {paths.algorithm}")
    if paths.run_name is not None:
        print(f"Run: {paths.run_name}")
    print(f"Experiment root: {paths.root_dir}")
    print(f"Results directory: {paths.results_dir}")
    print(f"Models directory: {paths.models_dir}")
