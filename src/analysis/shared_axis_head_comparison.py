"""Stage 5B+ comparison of source heads with four-dataset disease axes."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.training.reproducibility import git_identity, sha256_file

WEIGHTINGS = ("window_weighted", "subject_equal")


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("direction must be a finite vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("direction has zero norm")
    return vector / norm


def shared_axis(directions: np.ndarray) -> np.ndarray:
    """Return the normalized oriented Euclidean mean on the unit sphere."""
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2:
        raise ValueError("shared-axis directions must form a matrix")
    unit = np.stack([_unit(vector) for vector in directions])
    return _unit(unit.sum(axis=0))


def compare_head_to_directions(
    head_direction: np.ndarray, directions: np.ndarray, *, source_index: int
) -> dict:
    """Compute head-to-dataset and prototype-to-prototype summary geometry."""
    head = _unit(head_direction)
    directions = np.asarray(directions, dtype=np.float64)
    unit = np.stack([_unit(vector) for vector in directions])
    if not 0 <= source_index < len(unit):
        raise ValueError("source direction index is out of range")
    head_cosines = np.clip(unit @ head, -1.0, 1.0)
    pairwise = np.clip(unit @ unit.T, -1.0, 1.0)
    off_diagonal = pairwise[np.triu_indices(len(unit), 1)]
    cross_mask = np.arange(len(unit)) != source_index
    axis = shared_axis(unit)
    head_shared_cosine = float(np.clip(head @ axis, -1.0, 1.0))
    return {
        "head_to_dataset_cosines": head_cosines.tolist(),
        "head_to_dataset_mean": float(head_cosines.mean()),
        "head_to_cross_dataset_mean": float(head_cosines[cross_mask].mean()),
        "head_to_source_dataset_cosine": float(head_cosines[source_index]),
        "prototype_pairwise_mean": float(off_diagonal.mean()),
        "prototype_pairwise_min": float(off_diagonal.min()),
        "prototype_pairwise_max": float(off_diagonal.max()),
        "prototype_mean_minus_head_mean": float(
            off_diagonal.mean() - head_cosines.mean()
        ),
        "prototype_mean_minus_head_cross_dataset_mean": float(
            off_diagonal.mean() - head_cosines[cross_mask].mean()
        ),
        "shared_axis": axis.tolist(),
        "head_to_shared_axis_cosine": head_shared_cosine,
        "head_to_shared_axis_angle_degrees": float(
            math.degrees(math.acos(head_shared_cosine))
        ),
    }


def _validate_protocol(config: dict) -> None:
    if config.get("role") != "post_hoc_analysis":
        raise ValueError("Stage 5B+ requires post_hoc_analysis role")
    if config.get("target_label_usage") != "inherited frozen post-hoc artifacts only":
        raise ValueError("Stage 5B+ target-label declaration mismatch")
    if sha256_file(Path(config["protocol"])) != config.get("protocol_sha256"):
        raise ValueError("Stage 5B+ protocol hash mismatch")


def _load_reference(config: dict, reference: str) -> tuple[dict, dict, dict]:
    entry = config["references"][reference]
    stage5a_dir = Path(entry["stage5a_dir"])
    comparison_path = stage5a_dir / "direction_comparison.json"
    stage5a_manifest = _load_json(stage5a_dir / "run_manifest.json")
    if sha256_file(comparison_path) != stage5a_manifest.get(
        "direction_comparison_sha256"
    ):
        raise ValueError(f"Stage 5A comparison hash mismatch for {reference}")
    if stage5a_manifest.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5A head extraction was not label-free")

    stage5b_dir = Path(entry["stage5b_reference_dir"])
    summary_path = stage5b_dir / "feature_geometry_summary.json"
    stage5b_manifest = _load_json(stage5b_dir / "run_manifest.json")
    if sha256_file(summary_path) != stage5b_manifest.get("geometry_summary_sha256"):
        raise ValueError(f"Stage 5B summary hash mismatch for {reference}")
    if stage5b_manifest.get("selected_window_manifest_sha256") != config.get(
        "selected_window_manifest_sha256"
    ):
        raise ValueError("Stage 5B+ reference spaces used different cohorts")
    return (
        _load_json(comparison_path),
        _load_json(summary_path),
        {
            "stage5a_direction_comparison_sha256": sha256_file(comparison_path),
            "stage5b_geometry_summary_sha256": sha256_file(summary_path),
            "selected_window_manifest_sha256": stage5b_manifest[
                "selected_window_manifest_sha256"
            ],
        },
    )


def _plot(output_path: Path, names: list[str], results: dict) -> None:
    cache = Path(tempfile.gettempdir()) / "stage5b-plus-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    references = list(results)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, reference in enumerate(references):
        for column, weighting in enumerate(WEIGHTINGS):
            axis = axes[row, column]
            payload = results[reference][weighting]
            values = payload["head_to_dataset_cosines"]
            axis.bar(names, values, color="#4c78a8")
            axis.axhline(
                payload["prototype_pairwise_mean"],
                color="#e45756",
                linestyle="--",
                label="mean prototype-prototype",
            )
            axis.axhline(
                payload["head_to_shared_axis_cosine"],
                color="#54a24b",
                linestyle=":",
                label="head-to-shared-axis",
            )
            lower = min(0.8, min(values) - 0.02)
            axis.set_ylim(lower, 1.005)
            axis.set_title(f"{reference} reference — {weighting}")
            axis.set_ylabel("cosine similarity")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_head_vs_shared_axis(config: dict) -> dict:
    """Run the read-only Stage 5B+ artifact analysis."""
    config = deepcopy(config)
    _validate_protocol(config)
    names = list(config["dataset_order"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "head_to_dataset_directions.csv",
        "summary": output_dir / "head_to_shared_axis_summary.json",
        "plot": output_dir / "head_vs_shared_axis.png",
        "manifest": output_dir / "stage5b_plus_run_manifest.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Stage 5B+ outputs already exist: {existing}")
    results = {}
    provenance = {}
    csv_rows = []
    for reference in config["references"]:
        comparison, summary, hashes = _load_reference(config, reference)
        head = np.asarray(comparison["head_direction"], dtype=np.float64)
        source_index = names.index(reference)
        results[reference] = {}
        provenance[reference] = hashes
        for weighting in WEIGHTINGS:
            directions = np.asarray(
                [summary["datasets"][name][weighting]["direction"] for name in names]
            )
            payload = compare_head_to_directions(
                head, directions, source_index=source_index
            )
            results[reference][weighting] = payload
            for dataset, cosine in zip(names, payload["head_to_dataset_cosines"]):
                csv_rows.append(
                    {
                        "reference_model": reference,
                        "weighting": weighting,
                        "dataset_direction": dataset,
                        "is_source_dataset": dataset == reference,
                        "head_to_direction_cosine": cosine,
                        "angle_degrees": math.degrees(
                            math.acos(np.clip(cosine, -1, 1))
                        ),
                    }
                )
    with paths["csv"].with_suffix(".csv.tmp").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "reference_model",
            "weighting",
            "dataset_direction",
            "is_source_dataset",
            "head_to_direction_cosine",
            "angle_degrees",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    paths["csv"].with_suffix(".csv.tmp").replace(paths["csv"])
    result = {
        "role": "post_hoc_analysis",
        "new_target_labels_accessed": False,
        "target_label_usage": config["target_label_usage"],
        "dataset_order": names,
        "reference_spaces": results,
    }
    _write_json(paths["summary"], result)
    _plot(paths["plot"], names, results)
    manifest = {
        "git": git_identity(),
        "config": config,
        "new_target_labels_accessed": False,
        "input_provenance": provenance,
        "artifacts": {
            "head_to_dataset_directions_sha256": sha256_file(paths["csv"]),
            "head_to_shared_axis_summary_sha256": sha256_file(paths["summary"]),
            "head_vs_shared_axis_sha256": sha256_file(paths["plot"]),
        },
    }
    _write_json(paths["manifest"], manifest)
    return result
