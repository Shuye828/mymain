"""Stage 5B four-dataset disease-direction geometry analysis."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGWindowDataset, WindowRow, load_window_rows
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)

MANIFEST_FIELDS = (
    "dataset",
    "record_id",
    "subject_id",
    "start_sample",
    "end_sample",
    "fs_original",
    "binary_label",
    "rhythm_label",
    "source_split",
    "target_split",
)


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def validate_protocol(config: dict) -> None:
    if config.get("role") != "post_hoc_analysis":
        raise ValueError("Stage 5B requires role='post_hoc_analysis'")
    if config.get("target_label_usage") != "post-hoc mechanism analysis only":
        raise ValueError("Stage 5B target-label usage must be explicitly post-hoc")
    if sha256_file(Path(config["protocol"])) != config.get("protocol_sha256"):
        raise ValueError("Stage 5B protocol file hash mismatch")


def _row_as_dict(row: WindowRow) -> dict[str, str | int | float]:
    return {
        field: getattr(row, "start_sample" if field == "start_sample" else field)
        for field in MANIFEST_FIELDS
    }


def _parse_manifest_row(item: dict[str, str]) -> WindowRow:
    return WindowRow(
        dataset=item["dataset"],
        record_id=item["record_id"],
        subject_id=item["subject_id"],
        start_sample=int(item["start_sample"]),
        end_sample=int(item["end_sample"]),
        fs_original=float(item["fs_original"]),
        binary_label=int(item["binary_label"]),
        rhythm_label=item["rhythm_label"],
        source_split=item["source_split"],
        target_split=item["target_split"],
    )


def select_analysis_rows(config: dict) -> list[WindowRow]:
    """Select one deterministic, label-aware post-hoc cohort for both models."""
    validate_protocol(config)
    cap = int(config["selection"]["max_windows_per_subject_per_class"])
    rows: list[WindowRow] = []
    for dataset in config["dataset_order"]:
        entry = config["datasets"][dataset]
        selected = load_window_rows(
            [Path(entry["index_path"])],
            max_windows_per_subject_per_class=cap,
            seed=int(config["seed"]),
        )
        observed = {row.dataset for row in selected}
        if observed != {dataset}:
            raise ValueError(f"index dataset mismatch for {dataset}: {observed}")
        if {row.binary_label for row in selected} != {0, 1}:
            raise ValueError(f"dataset {dataset} does not contain both classes")
        rows.extend(selected)
    return rows


def prepare_selection(config: dict, *, output_override: Path | None = None) -> dict:
    """Freeze the common labelled analysis cohort before model extraction."""
    config = deepcopy(config)
    rows = select_analysis_rows(config)
    output_dir = output_override or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "selected_window_manifest.csv"
    artifact_path = output_dir / "selection_artifact.json"
    if manifest_path.exists() or artifact_path.exists():
        raise FileExistsError("Stage 5B selection has already been prepared")
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(_row_as_dict(row) for row in rows)
    temporary.replace(manifest_path)
    counts: dict[str, dict[str, int]] = {}
    for dataset in config["dataset_order"]:
        selected = [row for row in rows if row.dataset == dataset]
        counts[dataset] = {
            "windows": len(selected),
            "nonaf_windows": sum(row.binary_label == 0 for row in selected),
            "af_windows": sum(row.binary_label == 1 for row in selected),
            "subjects": len({row.subject_id for row in selected}),
        }
    artifact = {
        "frozen": True,
        "role": "post_hoc_analysis",
        "target_labels_accessed": True,
        "target_label_usage": config["target_label_usage"],
        "selection_rule": config["selection"],
        "counts": counts,
        "total_windows": len(rows),
        "index_sha256": {
            name: sha256_file(Path(config["datasets"][name]["index_path"]))
            for name in config["dataset_order"]
        },
        "selected_window_manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(artifact_path, artifact)
    return {**artifact, "output_dir": str(output_dir)}


def _load_selection(config: dict, output_dir: Path) -> tuple[list[WindowRow], dict]:
    manifest_path = output_dir / "selected_window_manifest.csv"
    artifact = _load_json(output_dir / "selection_artifact.json")
    if not artifact.get("frozen"):
        raise ValueError("Stage 5B selection is not frozen")
    if artifact.get("target_label_usage") != config["target_label_usage"]:
        raise ValueError("Stage 5B target-label declaration changed")
    if sha256_file(manifest_path) != artifact.get("selected_window_manifest_sha256"):
        raise ValueError("Stage 5B selected-window manifest hash mismatch")
    for name in config["dataset_order"]:
        if (
            sha256_file(Path(config["datasets"][name]["index_path"]))
            != artifact["index_sha256"][name]
        ):
            raise ValueError(f"Stage 5B input index changed for {name}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [_parse_manifest_row(item) for item in csv.DictReader(handle)]
    if len(rows) != artifact["total_windows"]:
        raise ValueError("Stage 5B selection row count mismatch")
    return rows, artifact


class GeometryAccumulator:
    """Streaming window- and subject-equal geometry over normalized features."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.class_sums = np.zeros((2, feature_dim), dtype=np.float64)
        self.class_counts = np.zeros(2, dtype=np.int64)
        self.global_sum = np.zeros(feature_dim, dtype=np.float64)
        self.count = 0
        self.subject_class_sums: dict[tuple[str, int], np.ndarray] = defaultdict(
            lambda: np.zeros(feature_dim, dtype=np.float64)
        )
        self.subject_class_counts: dict[tuple[str, int], int] = defaultdict(int)
        self.subject_sums: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(feature_dim, dtype=np.float64)
        )
        self.subject_counts: dict[str, int] = defaultdict(int)

    def update(
        self, features: np.ndarray, labels: np.ndarray, subjects: Iterable[str]
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        subjects = [str(value) for value in subjects]
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("feature batch has the wrong shape")
        if len(features) != len(labels) or len(features) != len(subjects):
            raise ValueError("geometry batch inputs are not aligned")
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid geometry batch")
        if not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5):
            raise ValueError("Stage 5B requires L2-normalized features")
        self.global_sum += features.sum(axis=0)
        self.count += len(features)
        for label in (0, 1):
            mask = labels == label
            self.class_sums[label] += features[mask].sum(axis=0)
            self.class_counts[label] += int(mask.sum())
        for feature, label, subject in zip(features, labels, subjects):
            key = (subject, int(label))
            self.subject_class_sums[key] += feature
            self.subject_class_counts[key] += 1
            self.subject_sums[subject] += feature
            self.subject_counts[subject] += 1

    @staticmethod
    def _unit_difference(af: np.ndarray, nonaf: np.ndarray) -> tuple[np.ndarray, float]:
        difference = af - nonaf
        norm = float(np.linalg.norm(difference))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("AF and non-AF prototypes define no direction")
        return difference / norm, norm

    def finalize(self) -> dict:
        if self.count == 0 or np.any(self.class_counts == 0):
            raise ValueError("geometry requires non-empty binary classes")
        prototypes = self.class_sums / self.class_counts[:, None]
        direction, direction_norm = self._unit_difference(prototypes[1], prototypes[0])
        subject_centroid = np.mean(
            [
                self.subject_sums[key] / self.subject_counts[key]
                for key in sorted(self.subject_sums)
            ],
            axis=0,
        )
        subject_prototypes = []
        subject_class_counts = []
        for label in (0, 1):
            keys = sorted(key for key in self.subject_class_sums if key[1] == label)
            subject_class_counts.append(len(keys))
            subject_prototypes.append(
                np.mean(
                    [
                        self.subject_class_sums[key] / self.subject_class_counts[key]
                        for key in keys
                    ],
                    axis=0,
                )
            )
        subject_direction, subject_direction_norm = self._unit_difference(
            subject_prototypes[1], subject_prototypes[0]
        )
        return {
            "window_count": int(self.count),
            "class_window_counts": self.class_counts.tolist(),
            "subject_count": len(self.subject_sums),
            "subject_class_counts": subject_class_counts,
            "window_weighted": {
                "centroid": (self.global_sum / self.count).tolist(),
                "nonaf_prototype": prototypes[0].tolist(),
                "af_prototype": prototypes[1].tolist(),
                "direction": direction.tolist(),
                "prototype_difference_norm": direction_norm,
            },
            "subject_equal": {
                "centroid": subject_centroid.tolist(),
                "nonaf_prototype": subject_prototypes[0].tolist(),
                "af_prototype": subject_prototypes[1].tolist(),
                "direction": subject_direction.tolist(),
                "prototype_difference_norm": subject_direction_norm,
            },
        }


def direction_cosine_matrix(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or not np.isfinite(directions).all():
        raise ValueError("directions must be a finite matrix")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("direction matrix contains a zero vector")
    unit = directions / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def centroid_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2 or not np.isfinite(centroids).all():
        raise ValueError("centroids must be a finite matrix")
    return np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)


def _balanced_diagnostic_rows(rows: list[WindowRow], cap: int) -> list[WindowRow]:
    """Keep a deterministic two-class diagnostic subset within a total cap."""
    per_class = cap // 2
    if per_class < 1:
        raise ValueError("diagnostic cap must allow at least one row per class")
    selected = []
    for label in (0, 1):
        selected.extend([row for row in rows if row.binary_label == label][:per_class])
    return sorted(
        selected, key=lambda row: (row.subject_id, row.record_id, row.start_sample)
    )


def extract_reference_geometry(
    config: dict,
    *,
    reference_name: str,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches_per_dataset: int | None = None,
) -> dict:
    """Extract four labelled dataset summaries in one frozen feature space."""
    config = deepcopy(config)
    validate_protocol(config)
    if reference_name not in config["references"]:
        raise ValueError(f"unknown Stage 5B reference {reference_name}")
    if max_batches_per_dataset is not None and max_batches_per_dataset <= 0:
        raise ValueError("diagnostic batch count must be positive")
    if max_batches_per_dataset is not None and output_override is None:
        raise ValueError("diagnostic extraction requires an output override")
    root = output_override or Path(config["output_dir"])
    rows, selection_artifact = _load_selection(config, root)
    entry = config["references"][reference_name]
    source_config = _load_json(Path(entry["source_config"]))
    if (
        source_config.get("role") != "source"
        or source_config.get("dataset") != reference_name
    ):
        raise ValueError("reference source configuration mismatch")
    checkpoint_path = Path(entry["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != entry.get("checkpoint_sha256"):
        raise ValueError(f"frozen checkpoint hash mismatch for {reference_name}")
    output_dir = root / f"{reference_name}_reference"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 5B reference output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    model = SourceMedTSTTT(**source_config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    feature_dim = int(source_config["model"]["dim"])
    extraction = config["extraction"]
    summaries: dict[str, dict] = {}
    started = time.perf_counter()
    for dataset_name in config["dataset_order"]:
        dataset_rows = [row for row in rows if row.dataset == dataset_name]
        if max_batches_per_dataset is not None:
            dataset_rows = _balanced_diagnostic_rows(
                dataset_rows, max_batches_per_dataset * int(extraction["batch_size"])
            )
        loader = DataLoader(
            ECGWindowDataset(
                dataset_rows,
                data_root=Path(config["datasets"][dataset_name]["data_root"]),
                expose_label=True,
            ),
            batch_size=int(extraction["batch_size"]),
            shuffle=False,
            num_workers=int(extraction["num_workers"]),
            pin_memory=False,
        )
        accumulator = GeometryAccumulator(feature_dim)
        dataset_started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            with torch.inference_mode():
                features = F.normalize(
                    model.forward_features(batch["x"].to(device)), dim=-1, eps=1e-12
                )
            accumulator.update(
                features.cpu().numpy(),
                batch["y"].numpy(),
                batch["metadata"]["subject_id"],
            )
            if batch_index % int(extraction["progress_every_batches"]) == 0:
                print(
                    f"phase=stage5b_extraction reference={reference_name} dataset={dataset_name} "
                    f"batch={batch_index} seconds={time.perf_counter() - dataset_started:.1f}",
                    flush=True,
                )
        summaries[dataset_name] = accumulator.finalize()

    summary_path = output_dir / "feature_geometry_summary.json"
    _write_json(
        summary_path,
        {
            "reference_model": reference_name,
            "representation": "backbone_l2",
            "target_labels_accessed": True,
            "target_label_usage": config["target_label_usage"],
            "diagnostic_max_batches_per_dataset": max_batches_per_dataset,
            "datasets": summaries,
        },
    )
    manifest = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "config": config,
        "reference_model": reference_name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_sha256,
        "selection_artifact_sha256": sha256_file(root / "selection_artifact.json"),
        "selected_window_manifest_sha256": selection_artifact[
            "selected_window_manifest_sha256"
        ],
        "geometry_summary_sha256": sha256_file(summary_path),
        "target_labels_accessed": True,
        "target_label_usage": config["target_label_usage"],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {
        "reference_model": reference_name,
        "counts": {name: value["window_count"] for name, value in summaries.items()},
        "output_dir": str(output_dir),
        "runtime_seconds": manifest["runtime_seconds"],
    }


def _write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", *names])
        for name, values in zip(names, matrix):
            writer.writerow([name, *[f"{value:.10f}" for value in values]])
    temporary.replace(path)


def _plot_heatmaps(
    root: Path, names: list[str], matrices: dict[str, np.ndarray]
) -> None:
    cache = Path(tempfile.gettempdir()) / "stage5b-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (key, matrix) in zip(axes.flat, matrices.items()):
        image = axis.imshow(matrix, cmap="coolwarm" if "cosine" in key else "viridis")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_title(key.replace("_", " ").title())
        threshold = matrix.mean()
        for row in range(len(names)):
            for column in range(len(names)):
                color = "white" if matrix[row, column] > threshold else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(root / "direction_and_centroid_heatmaps.png", dpi=180)
    plt.close(figure)


def finalize_geometry(config: dict, *, output_override: Path | None = None) -> dict:
    """Verify both reference summaries and produce matrices and heatmaps."""
    config = deepcopy(config)
    validate_protocol(config)
    root = output_override or Path(config["output_dir"])
    _, selection_artifact = _load_selection(config, root)
    names = list(config["dataset_order"])
    result: dict[str, Any] = {
        "role": "post_hoc_analysis",
        "target_labels_accessed": True,
        "target_label_usage": config["target_label_usage"],
        "dataset_order": names,
        "selection": selection_artifact,
        "reference_spaces": {},
    }
    for reference_name, entry in config["references"].items():
        reference_dir = root / f"{reference_name}_reference"
        summary_path = reference_dir / "feature_geometry_summary.json"
        manifest = _load_json(reference_dir / "run_manifest.json")
        if (
            manifest.get("selected_window_manifest_sha256")
            != selection_artifact["selected_window_manifest_sha256"]
        ):
            raise ValueError("reference spaces used different selected windows")
        if sha256_file(Path(entry["checkpoint"])) != manifest.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint changed for {reference_name}")
        if sha256_file(summary_path) != manifest.get("geometry_summary_sha256"):
            raise ValueError(f"geometry summary changed for {reference_name}")
        summary = _load_json(summary_path)
        if summary.get("diagnostic_max_batches_per_dataset") is not None:
            raise ValueError("cannot finalize diagnostic Stage 5B extraction")
        matrices: dict[str, np.ndarray] = {}
        for weighting in ("window_weighted", "subject_equal"):
            directions = np.asarray(
                [summary["datasets"][name][weighting]["direction"] for name in names]
            )
            centroids = np.asarray(
                [summary["datasets"][name][weighting]["centroid"] for name in names]
            )
            matrices[f"direction_cosine_{weighting}"] = direction_cosine_matrix(
                directions
            )
            matrices[f"centroid_distance_{weighting}"] = centroid_distance_matrix(
                centroids
            )
        for matrix_name, matrix in matrices.items():
            _write_matrix_csv(reference_dir / f"{matrix_name}.csv", names, matrix)
        _plot_heatmaps(reference_dir, names, matrices)
        result["reference_spaces"][reference_name] = {
            key: value.tolist() for key, value in matrices.items()
        }
    _write_json(root / "analysis_result.json", result)
    return {
        "dataset_order": names,
        "references": list(result["reference_spaces"]),
        "output_dir": str(root),
        "target_label_usage": config["target_label_usage"],
    }
