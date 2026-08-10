"""Stage 5D disease-axis distribution and boundary-shift analysis."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGWindowDataset, WindowRow
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import threshold_curve
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)

FORBIDDEN_SCORE_FIELDS = {"label", "labels", "binary_label", "rhythm_label"}


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


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_protocol(config: dict) -> None:
    if config.get("role") != "post_hoc_analysis":
        raise ValueError("Stage 5D requires post_hoc_analysis role")
    if config.get("target_label_usage") != "post-hoc mechanism analysis only":
        raise ValueError("Stage 5D target-label declaration mismatch")
    if sha256_file(Path(config["protocol"])) != config.get("protocol_sha256"):
        raise ValueError("Stage 5D protocol hash mismatch")


def load_hidden_selected_rows(path: Path) -> list[WindowRow]:
    """Load identity/signal fields while never parsing selected-manifest labels."""
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            rows.append(
                WindowRow(
                    dataset=item["dataset"],
                    record_id=item["record_id"],
                    subject_id=item["subject_id"],
                    start_sample=int(item["start_sample"]),
                    end_sample=int(item["end_sample"]),
                    fs_original=float(item["fs_original"]),
                    binary_label=-1,
                    rhythm_label="__HIDDEN_STAGE5D_LABEL__",
                    source_split=item["source_split"],
                    target_split=item["target_split"],
                )
            )
    return rows


def _write_score_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    forbidden = FORBIDDEN_SCORE_FIELDS & set(arrays)
    if forbidden:
        raise ValueError(
            f"Stage 5D score archive cannot contain labels: {sorted(forbidden)}"
        )
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("Stage 5D score arrays are not aligned")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def distribution_statistics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    bins: int = 200,
    score_range: tuple[float, float] = (-1.0, 1.0),
) -> dict:
    """Compute class-conditional axis geometry and empirical histogram overlap."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("distribution inputs must be aligned vectors")
    if np.unique(labels).size != 2 or not np.isin(labels, [0, 1]).all():
        raise ValueError("distribution statistics require both binary classes")
    if not np.isfinite(scores).all() or bins <= 1:
        raise ValueError("invalid distribution scores or bins")
    nonaf = scores[labels == 0]
    af = scores[labels == 1]
    mu_nonaf = float(nonaf.mean())
    mu_af = float(af.mean())
    std_nonaf = float(nonaf.std(ddof=0))
    std_af = float(af.std(ddof=0))
    gap = mu_af - mu_nonaf
    pooled_std = float(np.sqrt((std_nonaf**2 + std_af**2) / 2))
    if pooled_std <= 1e-12:
        raise ValueError("class distributions have zero pooled variance")
    h0, _ = np.histogram(nonaf, bins=bins, range=score_range)
    h1, _ = np.histogram(af, bins=bins, range=score_range)
    if h0.sum() != len(nonaf) or h1.sum() != len(af):
        raise ValueError("scores fall outside the frozen histogram range")
    overlap = float(np.minimum(h0 / h0.sum(), h1 / h1.sum()).sum())
    return {
        "support": len(labels),
        "nonaf_count": len(nonaf),
        "af_count": len(af),
        "af_prevalence": float(len(af) / len(labels)),
        "mu_nonaf": mu_nonaf,
        "mu_af": mu_af,
        "std_nonaf": std_nonaf,
        "std_af": std_af,
        "class_gap": gap,
        "midpoint": (mu_af + mu_nonaf) / 2,
        "pooled_std": pooled_std,
        "d_prime": gap / pooled_std,
        "histogram_overlap_coefficient": overlap,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "histogram_bins": bins,
        "histogram_range": list(score_range),
    }


def boundary_shift_statistics(
    dataset_stats: dict,
    source_stats: dict,
    *,
    p0_threshold: float,
    p1_threshold: float,
    oracle_threshold: float,
) -> dict:
    if abs(source_stats["class_gap"]) <= 1e-12:
        raise ValueError("source class gap is zero")
    return {
        "oracle_minus_p1": oracle_threshold - p1_threshold,
        "oracle_minus_p0": oracle_threshold - p0_threshold,
        "midpoint_drift": dataset_stats["midpoint"] - source_stats["midpoint"],
        "class_gap_ratio": dataset_stats["class_gap"] / source_stats["class_gap"],
    }


def _validate_reference(config: dict, reference: str) -> tuple[dict, dict, dict]:
    entry = config["references"][reference]
    source_config = _load_json(Path(entry["source_config"]))
    if (
        source_config.get("role") != "source"
        or source_config.get("dataset") != reference
    ):
        raise ValueError("Stage 5D source configuration mismatch")
    checkpoint = Path(entry["checkpoint"])
    direction_path = Path(entry["direction"])
    direction = _load_json(direction_path)
    direction_manifest = _load_json(direction_path.parent / "run_manifest.json")
    if direction_manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("Stage 5D checkpoint/direction provenance mismatch")
    if direction_manifest.get("diagnostic_max_batches") is not None:
        raise ValueError("Stage 5D rejects diagnostic source direction")
    threshold_path = Path(entry["stage5c_threshold_artifact"])
    threshold = _load_json(threshold_path)
    threshold_manifest = _load_json(threshold_path.parent / "selection_manifest.json")
    if sha256_file(threshold_path) != threshold_manifest.get(
        "threshold_artifact_sha256"
    ):
        raise ValueError("Stage 5D threshold artifact hash mismatch")
    if threshold.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5D source threshold selection was not label-isolated")
    return source_config, direction, threshold


def extract_axis_scores(
    config: dict,
    *,
    reference: str,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches_per_dataset: int | None = None,
) -> dict:
    """Freeze common-cohort disease-axis scores without parsing any labels."""
    config = deepcopy(config)
    _validate_protocol(config)
    source_config, direction, threshold = _validate_reference(config, reference)
    selection_path = Path(config["selected_window_manifest"])
    if sha256_file(selection_path) != config.get("selected_window_manifest_sha256"):
        raise ValueError("Stage 5D selected-window manifest hash mismatch")
    if max_batches_per_dataset is not None and max_batches_per_dataset <= 0:
        raise ValueError("diagnostic batch count must be positive")
    if max_batches_per_dataset is not None and output_override is None:
        raise ValueError("diagnostic extraction requires output override")
    rows = load_hidden_selected_rows(selection_path)
    if any(row.binary_label != -1 for row in rows):
        raise ValueError("Stage 5D extraction exposed labels")
    output_root = output_override or Path(config["output_dir"])
    output_dir = output_root / reference
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 5D output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    model = SourceMedTSTTT(**source_config["model"]).to(device)
    checkpoint_path = Path(config["references"][reference]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    direction_tensor = torch.tensor(
        direction["direction"], dtype=torch.float32, device=device
    )
    extraction = config["extraction"]
    parts = {
        key: []
        for key in ("dataset", "subject_id", "record_id", "window_start", "axis_score")
    }
    counts = {}
    started = time.perf_counter()
    for dataset_name in config["dataset_order"]:
        selected = [row for row in rows if row.dataset == dataset_name]
        if max_batches_per_dataset is not None:
            selected = selected[
                : max_batches_per_dataset * int(extraction["batch_size"])
            ]
        loader = DataLoader(
            ECGWindowDataset(
                selected, data_root=Path(config["data_root"]), expose_label=False
            ),
            batch_size=int(extraction["batch_size"]),
            shuffle=False,
            num_workers=int(extraction["num_workers"]),
            pin_memory=False,
        )
        dataset_count = 0
        dataset_started = time.perf_counter()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                if not bool((batch["y"] == -1).all()):
                    raise ValueError("Stage 5D loader exposed labels")
                features = F.normalize(
                    model.forward_features(batch["x"].to(device)), dim=-1, eps=1e-12
                )
                scores = features @ direction_tensor
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Stage 5D scoring produced NaN/Inf")
                metadata = batch["metadata"]
                parts["dataset"].extend(str(value) for value in metadata["dataset"])
                parts["subject_id"].extend(
                    str(value) for value in metadata["subject_id"]
                )
                parts["record_id"].extend(str(value) for value in metadata["record_id"])
                parts["window_start"].extend(
                    int(value) for value in metadata["window_start"].tolist()
                )
                parts["axis_score"].extend(
                    scores.cpu().numpy().astype(np.float32).tolist()
                )
                dataset_count += len(scores)
                if batch_index % int(extraction["progress_every_batches"]) == 0:
                    print(
                        f"phase=stage5d_scoring reference={reference} dataset={dataset_name} "
                        f"batch={batch_index} seconds={time.perf_counter()-dataset_started:.1f}",
                        flush=True,
                    )
        counts[dataset_name] = dataset_count
    arrays = {
        "dataset": np.asarray(parts["dataset"], dtype=np.str_),
        "subject_id": np.asarray(parts["subject_id"], dtype=np.str_),
        "record_id": np.asarray(parts["record_id"], dtype=np.str_),
        "window_start": np.asarray(parts["window_start"], dtype=np.int64),
        "axis_score": np.asarray(parts["axis_score"], dtype=np.float32),
    }
    score_path = output_dir / "axis_scores.npz"
    _write_score_archive(score_path, arrays)
    artifact = {
        "frozen": True,
        "reference_model": reference,
        "labels_accessed": False,
        "selected_manifest_label_fields_parsed": False,
        "diagnostic_max_batches_per_dataset": max_batches_per_dataset,
        "counts": counts,
        "selected_window_manifest_sha256": config["selected_window_manifest_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "direction_sha256": sha256_file(
            Path(config["references"][reference]["direction"])
        ),
        "stage5c_threshold_artifact_sha256": sha256_file(
            Path(config["references"][reference]["stage5c_threshold_artifact"])
        ),
        "score_sha256": sha256_file(score_path),
    }
    artifact_path = output_dir / "score_artifact.json"
    _write_json(artifact_path, artifact)
    manifest = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "config": config,
        "reference_model": reference,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "labels_accessed": False,
        "score_artifact_sha256": sha256_file(artifact_path),
        "runtime_seconds": time.perf_counter() - started,
        "p0_threshold": threshold["thresholds"]["P0"]["threshold"],
        "p1_threshold": threshold["thresholds"]["P1"]["threshold"],
    }
    _write_json(output_dir / "extraction_manifest.json", manifest)
    return {
        "reference_model": reference,
        "labels_accessed": False,
        "counts": counts,
        "runtime_seconds": manifest["runtime_seconds"],
        "output_dir": str(output_dir),
    }


def _selected_label_map(path: Path) -> dict:
    mapping = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            key = (
                item["dataset"],
                item["subject_id"],
                item["record_id"],
                int(item["start_sample"]),
            )
            if key in mapping:
                raise ValueError(f"duplicate Stage 5D selected key {key}")
            label = int(item["binary_label"])
            if label not in (0, 1):
                raise ValueError("Stage 5D selected manifest has invalid label")
            mapping[key] = label
    return mapping


def _join_labels(archive: np.lib.npyio.NpzFile, mapping: dict) -> np.ndarray:
    datasets = archive["dataset"]
    subjects = archive["subject_id"]
    records = archive["record_id"]
    starts = archive["window_start"]
    labels = []
    seen = set()
    for dataset, subject, record, start in zip(datasets, subjects, records, starts):
        key = (str(dataset), str(subject), str(record), int(start))
        if key in seen or key not in mapping:
            raise ValueError(f"invalid or duplicate frozen Stage 5D key {key}")
        seen.add(key)
        labels.append(mapping[key])
    return np.asarray(labels, dtype=np.int64)


def _plot_distributions(
    path: Path,
    datasets: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    source_name: str,
    target_names: list[str],
    p1_threshold: float,
    oracle_thresholds: dict[str, float],
    bins: int,
    score_range: tuple[float, float],
) -> None:
    cache = Path(tempfile.gettempdir()) / "stage5d-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(target_names),
        1,
        figsize=(12, 4 * len(target_names)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = {
        ("source", 0): "#4c78a8",
        ("source", 1): "#f58518",
        ("target", 0): "#72b7b2",
        ("target", 1): "#e45756",
    }
    for axis, target in zip(axes, target_names):
        for domain, dataset_name in (("source", source_name), ("target", target)):
            for label, class_name in ((0, "non-AF"), (1, "AF")):
                values = scores[(datasets == dataset_name) & (labels == label)]
                axis.hist(
                    values,
                    bins=bins,
                    range=score_range,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=colors[(domain, label)],
                    label=f"{dataset_name} {class_name}",
                )
        axis.axvline(p1_threshold, color="black", linestyle="--", label="source-val P1")
        axis.axvline(
            oracle_thresholds[target],
            color="#b279a2",
            linestyle=":",
            linewidth=2,
            label="target oracle (post-hoc)",
        )
        axis.set_title(f"{source_name} source vs {target}")
        axis.set_xlabel("source disease-axis score")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


STAT_FIELDS = [
    "reference_model",
    "dataset",
    "is_source_dataset",
    "support",
    "nonaf_count",
    "af_count",
    "af_prevalence",
    "mu_nonaf",
    "mu_af",
    "std_nonaf",
    "std_af",
    "class_gap",
    "midpoint",
    "pooled_std",
    "d_prime",
    "histogram_overlap_coefficient",
    "auroc",
    "auprc",
    "p0_threshold",
    "p1_threshold",
    "oracle_threshold",
    "oracle_balanced_accuracy",
    "oracle_minus_p1",
    "oracle_minus_p0",
    "midpoint_drift",
    "class_gap_ratio",
]


def finalize_axis_distribution(
    config: dict, *, reference: str, output_override: Path | None = None
) -> dict:
    """Join labels after score freeze and quantify distribution/boundary shift."""
    config = deepcopy(config)
    _validate_protocol(config)
    _, _, threshold = _validate_reference(config, reference)
    output_root = output_override or Path(config["output_dir"])
    output_dir = output_root / reference
    score_path = output_dir / "axis_scores.npz"
    artifact_path = output_dir / "score_artifact.json"
    extraction_path = output_dir / "extraction_manifest.json"
    artifact = _load_json(artifact_path)
    extraction = _load_json(extraction_path)
    if not artifact.get("frozen") or artifact.get("labels_accessed") is not False:
        raise ValueError("Stage 5D scores were not frozen label-free")
    if artifact.get("selected_manifest_label_fields_parsed") is not False:
        raise ValueError("Stage 5D extraction parsed selected-manifest labels")
    if extraction.get("labels_accessed") is not False:
        raise ValueError("Stage 5D extraction manifest is not label-free")
    if artifact.get("diagnostic_max_batches_per_dataset") is not None:
        raise ValueError("Stage 5D formal finalize rejects diagnostic scores")
    if sha256_file(score_path) != artifact.get("score_sha256"):
        raise ValueError("Stage 5D score hash mismatch")
    if sha256_file(artifact_path) != extraction.get("score_artifact_sha256"):
        raise ValueError("Stage 5D score artifact hash mismatch")
    selection_path = Path(config["selected_window_manifest"])
    if sha256_file(selection_path) != artifact.get("selected_window_manifest_sha256"):
        raise ValueError("Stage 5D selected manifest changed after score freeze")
    archive = np.load(score_path)
    forbidden = FORBIDDEN_SCORE_FIELDS & set(archive.files)
    if forbidden:
        raise ValueError(f"Stage 5D score archive leaked labels: {sorted(forbidden)}")
    mapping = _selected_label_map(selection_path)
    labels = _join_labels(archive, mapping)
    if len(labels) != len(mapping):
        raise ValueError("Stage 5D formal score archive does not cover full cohort")
    datasets = archive["dataset"]
    scores = archive["axis_score"].astype(np.float64)
    bins = int(config["statistics"]["histogram_bins"])
    score_range = tuple(float(value) for value in config["statistics"]["score_range"])
    p0 = float(threshold["thresholds"]["P0"]["threshold"])
    p1 = float(threshold["thresholds"]["P1"]["threshold"])
    statistics = {}
    oracle = {}
    for dataset_name in config["dataset_order"]:
        mask = datasets == dataset_name
        statistics[dataset_name] = distribution_statistics(
            labels[mask], scores[mask], bins=bins, score_range=score_range
        )
        _, selected = threshold_curve(labels[mask], scores[mask], fixed_threshold=p1)
        oracle[dataset_name] = selected
    source_stats = statistics[reference]
    rows = []
    operating = {}
    for dataset_name in config["dataset_order"]:
        mask = datasets == dataset_name
        selected = oracle[dataset_name]
        shift = boundary_shift_statistics(
            statistics[dataset_name],
            source_stats,
            p0_threshold=p0,
            p1_threshold=p1,
            oracle_threshold=float(selected["threshold"]),
        )
        rows.append(
            {
                "reference_model": reference,
                "dataset": dataset_name,
                "is_source_dataset": dataset_name == reference,
                **{
                    key: value
                    for key, value in statistics[dataset_name].items()
                    if key not in {"histogram_bins", "histogram_range"}
                },
                "p0_threshold": p0,
                "p1_threshold": p1,
                "oracle_threshold": selected["threshold"],
                "oracle_balanced_accuracy": selected["balanced_accuracy"],
                **shift,
            }
        )
        operating[dataset_name] = {
            "P0": compute_binary_metrics(labels[mask], scores[mask], threshold=p0),
            "P1": compute_binary_metrics(labels[mask], scores[mask], threshold=p1),
            "oracle_post_hoc_only": compute_binary_metrics(
                labels[mask], scores[mask], threshold=float(selected["threshold"])
            ),
        }
    stats_path = output_dir / "distribution_statistics.csv"
    _write_csv(stats_path, rows, STAT_FIELDS)
    result = {
        "reference_model": reference,
        "role": "post_hoc_analysis",
        "target_label_usage": "post-hoc mechanism analysis only",
        "oracle_threshold_usage": "mechanism analysis only; prohibited for adaptation/model selection",
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_analysis_labels_accessed": True,
        "p0_threshold": p0,
        "p1_threshold": p1,
        "dataset_statistics": statistics,
        "oracle_thresholds": oracle,
        "operating_metrics": operating,
        "boundary_shift_rows": rows,
    }
    result_path = output_dir / "analysis_result.json"
    _write_json(result_path, result)
    plot_path = output_dir / "axis_distribution_shift.png"
    _plot_distributions(
        plot_path,
        datasets,
        labels,
        scores,
        source_name=reference,
        target_names=[name for name in config["dataset_order"] if name != reference],
        p1_threshold=p1,
        oracle_thresholds={
            name: float(value["threshold"]) for name, value in oracle.items()
        },
        bins=bins,
        score_range=score_range,
    )
    evaluation_manifest = {
        "extraction_git": extraction["git"],
        "analysis_git": git_identity(),
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_analysis_labels_accessed": True,
        "oracle_threshold_usage": result["oracle_threshold_usage"],
        "score_sha256": artifact["score_sha256"],
        "distribution_statistics_sha256": sha256_file(stats_path),
        "analysis_result_sha256": sha256_file(result_path),
        "figure_sha256": sha256_file(plot_path),
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return result


def summarize_axis_distribution(config: dict) -> dict:
    config = deepcopy(config)
    _validate_protocol(config)
    output_dir = Path(config["output_dir"])
    rows = []
    references = {}
    for reference in config["references"]:
        source_dir = output_dir / reference
        result_path = source_dir / "analysis_result.json"
        manifest = _load_json(source_dir / "evaluation_manifest.json")
        if sha256_file(result_path) != manifest.get("analysis_result_sha256"):
            raise ValueError(f"Stage 5D result hash mismatch for {reference}")
        result = _load_json(result_path)
        references[reference] = result
        rows.extend(result["boundary_shift_rows"])
    combined_path = output_dir / "distribution_statistics.csv"
    _write_csv(combined_path, rows, STAT_FIELDS)
    summary = {
        "experiment": config["experiment"],
        "role": "post_hoc_analysis",
        "adaptation_time_target_labels_accessed": False,
        "final_transfer_results_used_for_model_selection": False,
        "oracle_threshold_usage": "mechanism analysis only; prohibited for adaptation/model selection",
        "references": references,
        "distribution_statistics_sha256": sha256_file(combined_path),
    }
    _write_json(output_dir / "analysis_result.json", summary)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "git": git_identity(),
            "config": config,
            "distribution_statistics_sha256": summary["distribution_statistics_sha256"],
        },
    )
    return summary
