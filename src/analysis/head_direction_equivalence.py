"""Stage 5A comparison of prototype and binary-classifier directions."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    ECGWindowDataset,
    WindowRow,
    load_unlabeled_target_rows,
    load_window_rows,
)
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)

SCOPES = ("source_validation", "source_test", "target_evaluation")
SCORE_FIELDS = ("prototype_score", "classifier_logit_difference")
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


def _write_score_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    forbidden = FORBIDDEN_SCORE_FIELDS & set(arrays)
    if forbidden:
        raise ValueError(f"score archive cannot contain labels: {sorted(forbidden)}")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("score archive arrays are not aligned")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def compare_directions(
    prototype_direction: np.ndarray,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
) -> dict[str, Any]:
    """Compare normalized non-AF-to-AF prototype and linear-head axes."""

    prototype = np.asarray(prototype_direction, dtype=np.float64)
    weight = np.asarray(head_weight, dtype=np.float64)
    bias = np.asarray(head_bias, dtype=np.float64)
    if prototype.ndim != 1 or not np.isfinite(prototype).all():
        raise ValueError("prototype direction must be a finite vector")
    if weight.shape != (2, prototype.size) or bias.shape != (2,):
        raise ValueError("binary head shapes do not match prototype direction")
    head_difference = weight[1] - weight[0]
    head_norm = float(np.linalg.norm(head_difference))
    prototype_norm = float(np.linalg.norm(prototype))
    if head_norm <= 1e-12 or prototype_norm <= 1e-12:
        raise ValueError("direction vector has zero norm")
    normalized_prototype = prototype / prototype_norm
    normalized_head = head_difference / head_norm
    cosine = float(np.clip(normalized_prototype @ normalized_head, -1.0, 1.0))
    return {
        "feature_dim": int(prototype.size),
        "prototype_direction_norm": prototype_norm,
        "head_weight_difference_norm": head_norm,
        "head_bias_difference": float(bias[1] - bias[0]),
        "cosine": cosine,
        "angle_degrees": float(math.degrees(math.acos(cosine))),
        "head_direction": normalized_head.tolist(),
        "classifier_score_definition": "logit_AF_minus_logit_nonAF",
        "prototype_score_definition": "l2_backbone_feature_dot_d_proto",
    }


def compute_score_correlations(
    scopes: np.ndarray,
    prototype_scores: np.ndarray,
    classifier_scores: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Compute Pearson and Spearman correlations for every frozen split."""

    scopes = np.asarray(scopes)
    prototype_scores = np.asarray(prototype_scores, dtype=np.float64)
    classifier_scores = np.asarray(classifier_scores, dtype=np.float64)
    if not (
        scopes.ndim
        == prototype_scores.ndim
        == classifier_scores.ndim
        == 1
    ):
        raise ValueError("correlation inputs must be one-dimensional")
    if not (len(scopes) == len(prototype_scores) == len(classifier_scores)):
        raise ValueError("correlation inputs are not aligned")
    if not (
        np.isfinite(prototype_scores).all()
        and np.isfinite(classifier_scores).all()
    ):
        raise ValueError("score arrays contain NaN or Inf")
    output: dict[str, dict[str, float | int]] = {}
    for scope in SCOPES:
        mask = scopes == scope
        if int(mask.sum()) < 2:
            raise ValueError(f"scope {scope} has fewer than two scores")
        pearson = pearsonr(prototype_scores[mask], classifier_scores[mask])
        spearman = spearmanr(prototype_scores[mask], classifier_scores[mask])
        if not np.isfinite([pearson.statistic, spearman.statistic]).all():
            raise ValueError(f"scope {scope} has undefined score correlation")
        output[scope] = {
            "support": int(mask.sum()),
            "pearson": float(pearson.statistic),
            "spearman": float(spearman.statistic),
        }
    return output


def compute_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    allow_single_class: bool = False,
) -> dict:
    """Compute label-based ranking metrics without selecting a threshold."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("ranking labels and scores must be aligned vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("ranking labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("ranking scores contain NaN or Inf")
    base = {
        "support": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
    }
    if np.unique(labels).size != 2:
        if not allow_single_class:
            raise ValueError("ranking metrics require both binary classes")
        return {**base, "auroc": None, "auprc": None}
    return {
        **base,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def compute_stage5a_scores(
    logits: torch.Tensor,
    raw_features: torch.Tensor,
    prototype_direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized prototype score and exact binary logit difference."""

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Stage 5A requires binary [B,2] logits")
    if raw_features.ndim != 2 or raw_features.shape[0] != logits.shape[0]:
        raise ValueError("Stage 5A logits and features are not aligned")
    if prototype_direction.shape != (raw_features.shape[1],):
        raise ValueError("prototype direction has the wrong feature dimension")
    if not (
        torch.isfinite(logits).all()
        and torch.isfinite(raw_features).all()
        and torch.isfinite(prototype_direction).all()
    ):
        raise FloatingPointError("Stage 5A inputs contain NaN or Inf")
    normalized_features = F.normalize(raw_features, dim=-1, eps=1e-12)
    prototype_score = normalized_features @ prototype_direction
    classifier_score = logits[:, 1] - logits[:, 0]
    if not (
        torch.isfinite(prototype_score).all()
        and torch.isfinite(classifier_score).all()
    ):
        raise FloatingPointError("Stage 5A scoring produced NaN or Inf")
    return prototype_score, classifier_score


def _validate_protocol(config: dict) -> None:
    if config.get("role") != "post_hoc_analysis":
        raise ValueError("Stage 5A requires role='post_hoc_analysis'")
    protocol_path = Path(config["protocol"])
    if sha256_file(protocol_path) != config.get("protocol_sha256"):
        raise ValueError("Stage 5A protocol file hash mismatch")


def _source_entry(config: dict, source_name: str) -> dict:
    try:
        return deepcopy(config["sources"][source_name])
    except KeyError as exc:
        raise ValueError(f"unknown Stage 5A source {source_name}") from exc


def _validate_inputs(
    config: dict,
    source_name: str,
) -> tuple[dict, dict, dict, dict]:
    _validate_protocol(config)
    entry = _source_entry(config, source_name)
    source_config_path = Path(entry["source_config"])
    source_config = _load_json(source_config_path)
    if source_config.get("role") != "source":
        raise ValueError("referenced checkpoint configuration is not source-only")
    if source_config.get("dataset") != source_name:
        raise ValueError("source configuration dataset mismatch")
    checkpoint_path = Path(entry["checkpoint"])
    direction_path = Path(entry["direction"])
    direction = _load_json(direction_path)
    direction_manifest = _load_json(direction_path.parent / "run_manifest.json")
    if direction.get("dataset") != source_name:
        raise ValueError("prototype direction dataset mismatch")
    if direction.get("representation", {}).get("kind") != "backbone_l2":
        raise ValueError("Stage 5A requires the frozen backbone_l2 direction")
    if direction_manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("checkpoint does not match direction provenance")
    if direction_manifest.get("diagnostic_max_batches") is not None:
        raise ValueError("formal direction artifact is diagnostic")
    source_index = Path(source_config["index_path"])
    if direction_manifest.get("index_sha256") != sha256_file(source_index):
        raise ValueError("source index does not match direction provenance")
    target_index = Path(entry["target_index"])
    if not target_index.is_file():
        raise FileNotFoundError(target_index)
    return entry, source_config, direction, direction_manifest


def _hidden_source_rows(index_path: Path, split: str) -> list[WindowRow]:
    visible = load_window_rows([index_path], source_split=split)
    return [
        replace(
            row,
            binary_label=-1,
            rhythm_label="__HIDDEN_ANALYSIS_LABEL__",
        )
        for row in visible
    ]


def _score_rows(
    model: SourceMedTSTTT,
    rows: list[WindowRow],
    *,
    data_root: Path,
    scope: str,
    prototype_direction: torch.Tensor,
    device: torch.device,
    extraction: dict,
) -> dict[str, list | np.ndarray]:
    dataset = ECGWindowDataset(rows, data_root=data_root, expose_label=False)
    loader = DataLoader(
        dataset,
        batch_size=int(extraction["batch_size"]),
        shuffle=False,
        num_workers=int(extraction["num_workers"]),
        pin_memory=False,
    )
    progress_every = int(extraction.get("progress_every_batches", 250))
    if progress_every <= 0:
        raise ValueError("progress interval must be positive")
    datasets: list[str] = []
    subjects: list[str] = []
    records: list[str] = []
    starts: list[int] = []
    prototype_chunks: list[np.ndarray] = []
    classifier_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if not bool((batch["y"] == -1).all()):
                raise ValueError("score extraction exposed analysis labels")
            logits, raw_features = model(
                batch["x"].to(device), return_features=True
            )
            prototype_score, classifier_score = compute_stage5a_scores(
                logits, raw_features, prototype_direction
            )
            prototype_chunks.append(
                prototype_score.cpu().numpy().astype(np.float32)
            )
            classifier_chunks.append(
                classifier_score.cpu().numpy().astype(np.float32)
            )
            metadata = batch["metadata"]
            datasets.extend(str(value) for value in metadata["dataset"])
            subjects.extend(str(value) for value in metadata["subject_id"])
            records.extend(str(value) for value in metadata["record_id"])
            starts.extend(
                int(value) for value in metadata["window_start"].tolist()
            )
            if batch_index % progress_every == 0:
                print(
                    f"phase=stage5a_scoring scope={scope} batch={batch_index} "
                    f"samples={len(starts)} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    return {
        "dataset": datasets,
        "subject_id": subjects,
        "record_id": records,
        "window_start": starts,
        "analysis_scope": [scope] * len(starts),
        "prototype_score": np.concatenate(prototype_chunks),
        "classifier_logit_difference": np.concatenate(classifier_chunks),
    }


def _merge_score_parts(parts: list[dict[str, list | np.ndarray]]) -> dict:
    arrays: dict[str, np.ndarray] = {}
    for key in (
        "dataset",
        "subject_id",
        "record_id",
        "analysis_scope",
    ):
        arrays[key] = np.asarray(
            [value for part in parts for value in part[key]], dtype=np.str_
        )
    arrays["window_start"] = np.asarray(
        [value for part in parts for value in part["window_start"]],
        dtype=np.int64,
    )
    for key in SCORE_FIELDS:
        arrays[key] = np.concatenate([np.asarray(part[key]) for part in parts])
    return arrays


def _equivalence_status(
    direction_cosine: float,
    correlations: dict,
    rule: dict,
) -> dict:
    min_spearman = min(
        float(correlations[scope]["spearman"]) for scope in SCOPES
    )
    cosine_pass = direction_cosine > float(rule["min_direction_cosine"])
    spearman_pass = min_spearman > float(rule["min_split_spearman"])
    if cosine_pass and spearman_pass:
        conclusion = "highly_equivalent"
    elif cosine_pass or spearman_pass:
        conclusion = "partially_equivalent"
    else:
        conclusion = "clearly_different"
    return {
        "conclusion": conclusion,
        "minimum_split_spearman": min_spearman,
        "direction_cosine_pass": cosine_pass,
        "all_split_spearman_pass": spearman_pass,
        "rule": rule,
        "uses_labels": False,
    }


def extract_head_direction_scores(
    config: dict,
    *,
    source_name: str,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches_per_split: int | None = None,
) -> dict:
    """Freeze exact scores while keeping target labels inaccessible."""

    config = deepcopy(config)
    entry, source_config, direction, direction_manifest = _validate_inputs(
        config, source_name
    )
    extraction = config["extraction"]
    batch_size = int(extraction["batch_size"])
    if batch_size <= 0:
        raise ValueError("extraction batch size must be positive")
    if max_batches_per_split is not None and max_batches_per_split <= 0:
        raise ValueError("diagnostic batch count must be positive")
    if max_batches_per_split is not None and output_override is None:
        raise ValueError("diagnostic extraction requires an output override")
    output_dir = output_override or Path(entry["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Stage 5A output directory is not empty: {output_dir}"
        )

    source_index = Path(source_config["index_path"])
    target_index = Path(entry["target_index"])
    rows_by_scope = {
        "source_validation": _hidden_source_rows(source_index, "validation"),
        "source_test": _hidden_source_rows(source_index, "test"),
        "target_evaluation": load_unlabeled_target_rows(
            [target_index], target_split="evaluation"
        ),
    }
    full_counts = {scope: len(rows) for scope, rows in rows_by_scope.items()}
    expected_target = entry["target_dataset"]
    target_datasets = {row.dataset for row in rows_by_scope["target_evaluation"]}
    if target_datasets != {expected_target}:
        raise ValueError("target evaluation index dataset mismatch")
    if max_batches_per_split is not None:
        cap = max_batches_per_split * batch_size
        rows_by_scope = {
            scope: rows[:cap] for scope, rows in rows_by_scope.items()
        }
    if any(len(rows) < 2 for rows in rows_by_scope.values()):
        raise ValueError("every Stage 5A scope requires at least two windows")

    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    model = SourceMedTSTTT(**source_config["model"]).to(device)
    checkpoint_path = Path(entry["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    prototype = np.asarray(direction["direction"], dtype=np.float64)
    comparison = compare_directions(
        prototype,
        model.backbone.classification_head.weight.detach().cpu().numpy(),
        model.backbone.classification_head.bias.detach().cpu().numpy(),
    )
    prototype_tensor = torch.tensor(
        prototype, dtype=torch.float32, device=device
    )

    parts: list[dict[str, list | np.ndarray]] = []
    started = time.perf_counter()
    for scope in SCOPES:
        data_root = (
            Path(entry["target_data_root"])
            if scope == "target_evaluation"
            else Path(source_config["data_root"])
        )
        parts.append(
            _score_rows(
                model,
                rows_by_scope[scope],
                data_root=data_root,
                scope=scope,
                prototype_direction=prototype_tensor,
                device=device,
                extraction=extraction,
            )
        )
    arrays = _merge_score_parts(parts)
    correlations = compute_score_correlations(
        arrays["analysis_scope"],
        arrays["prototype_score"],
        arrays["classifier_logit_difference"],
    )
    equivalence = _equivalence_status(
        comparison["cosine"], correlations, config["equivalence_rule"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.npz"
    comparison_path = output_dir / "direction_comparison.json"
    correlation_path = output_dir / "score_correlation.json"
    artifact_path = output_dir / "score_artifact.json"
    manifest_path = output_dir / "run_manifest.json"
    _write_score_archive(score_path, arrays)
    _write_json(comparison_path, comparison)
    _write_json(
        correlation_path,
        {"correlations": correlations, "equivalence": equivalence},
    )
    selected_counts = {
        scope: len(rows) for scope, rows in rows_by_scope.items()
    }
    artifact = {
        "frozen": True,
        "target_labels_accessed": False,
        "target_label_usage": "post-hoc analysis only after score freeze",
        "source_dataset": source_name,
        "target_dataset": expected_target,
        "scopes": list(SCOPES),
        "full_counts": full_counts,
        "selected_counts": selected_counts,
        "diagnostic_max_batches_per_split": max_batches_per_split,
        "source_index_sha256": sha256_file(source_index),
        "target_index_sha256": sha256_file(target_index),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "direction_sha256": sha256_file(Path(entry["direction"])),
        "score_sha256": sha256_file(score_path),
    }
    _write_json(artifact_path, artifact)
    manifest = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "config": config,
        "source_entry": entry,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "direction_manifest": direction_manifest,
        "target_labels_accessed": False,
        "score_artifact_sha256": sha256_file(artifact_path),
        "direction_comparison_sha256": sha256_file(comparison_path),
        "score_correlation_sha256": sha256_file(correlation_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    return {
        "source_dataset": source_name,
        "target_dataset": expected_target,
        "target_labels_accessed": False,
        "counts": selected_counts,
        "direction_cosine": comparison["cosine"],
        "angle_degrees": comparison["angle_degrees"],
        "minimum_split_spearman": equivalence["minimum_split_spearman"],
        "equivalence": equivalence["conclusion"],
        "output_dir": str(output_dir),
        "runtime_seconds": manifest["runtime_seconds"],
    }


def _label_map(config: dict, entry: dict, source_config: dict) -> dict:
    mapping: dict[tuple[str, str, str, int], tuple[int, str]] = {}
    source_index = Path(source_config["index_path"])
    for split, scope in (
        ("validation", "source_validation"),
        ("test", "source_test"),
    ):
        for row in load_window_rows([source_index], source_split=split):
            key = (row.dataset, row.subject_id, row.record_id, row.start_sample)
            if key in mapping:
                raise ValueError(f"duplicate source analysis key {key}")
            mapping[key] = (row.binary_label, scope)
    for row in load_window_rows(
        [Path(entry["target_index"])], target_split="evaluation"
    ):
        key = (row.dataset, row.subject_id, row.record_id, row.start_sample)
        if key in mapping:
            raise ValueError(f"duplicate target analysis key {key}")
        mapping[key] = (row.binary_label, "target_evaluation")
    return mapping


def _join_labels(archive: np.lib.npyio.NpzFile, mapping: dict) -> np.ndarray:
    labels: list[int] = []
    seen: set[tuple[str, str, str, int]] = set()
    for dataset, subject, record, start, scope in zip(
        archive["dataset"].tolist(),
        archive["subject_id"].tolist(),
        archive["record_id"].tolist(),
        archive["window_start"].tolist(),
        archive["analysis_scope"].tolist(),
    ):
        key = (str(dataset), str(subject), str(record), int(start))
        if key in seen:
            raise ValueError(f"duplicate frozen score key {key}")
        seen.add(key)
        if key not in mapping:
            raise ValueError(f"frozen score key absent from current index: {key}")
        label, expected_scope = mapping[key]
        if str(scope) != expected_scope:
            raise ValueError(f"analysis scope mismatch for {key}")
        labels.append(label)
    return np.asarray(labels, dtype=np.int64)


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "source_dataset",
        "evaluation_dataset",
        "analysis_scope",
        "score_type",
        "support",
        "positive_count",
        "negative_count",
        "auroc",
        "auprc",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _plot_results(
    output_dir: Path,
    archive: np.lib.npyio.NpzFile,
    labels: np.ndarray,
    correlations: dict,
) -> None:
    cache = Path(tempfile.gettempdir()) / "stage5a-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scopes = archive["analysis_scope"]
    prototype = archive["prototype_score"].astype(np.float64)
    classifier = archive["classifier_logit_difference"].astype(np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, scope in zip(axes, SCOPES):
        indices = np.flatnonzero(scopes == scope)
        if len(indices) > 20000:
            indices = indices[
                np.linspace(0, len(indices) - 1, 20000, dtype=np.int64)
            ]
        axis.scatter(
            prototype[indices], classifier[indices], s=3, alpha=0.15
        )
        axis.set_title(
            f"{scope}\n"
            f"r={correlations[scope]['pearson']:.4f}, "
            f"rho={correlations[scope]['spearman']:.4f}"
        )
        axis.set_xlabel("prototype score")
        axis.set_ylabel("head logit difference")
        axis.grid(alpha=0.2)
    figure.savefig(output_dir / "score_scatter.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    for row_index, scope in enumerate(SCOPES):
        mask = scopes == scope
        for column, (field, title) in enumerate(
            (
                (prototype, "prototype score"),
                (classifier, "head logit difference"),
            )
        ):
            axis = axes[row_index, column]
            for label, name in ((0, "non-AF"), (1, "AF")):
                values = field[mask & (labels == label)]
                if values.size == 0:
                    continue
                axis.hist(
                    values,
                    bins=80,
                    density=True,
                    alpha=0.45,
                    label=name,
                )
            axis.set_title(f"{scope}: {title}")
            axis.set_ylabel("density")
            axis.legend()
            axis.grid(alpha=0.2)
    figure.savefig(output_dir / "score_histograms.png", dpi=180)
    plt.close(figure)


def finalize_head_direction_analysis(
    config: dict,
    *,
    source_name: str,
    output_override: Path | None = None,
) -> dict:
    """Join labels only after verifying immutable Stage 5A score artifacts."""

    config = deepcopy(config)
    entry, source_config, _, _ = _validate_inputs(config, source_name)
    output_dir = output_override or Path(entry["output_dir"])
    score_path = output_dir / "scores.npz"
    artifact_path = output_dir / "score_artifact.json"
    manifest_path = output_dir / "run_manifest.json"
    comparison_path = output_dir / "direction_comparison.json"
    correlation_path = output_dir / "score_correlation.json"
    artifact = _load_json(artifact_path)
    manifest = _load_json(manifest_path)
    comparison = _load_json(comparison_path)
    correlation_payload = _load_json(correlation_path)
    if not artifact.get("frozen") or artifact.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5A scores were not frozen label-free")
    if manifest.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5A manifest does not prove label-free scoring")
    if sha256_file(score_path) != artifact.get("score_sha256"):
        raise ValueError("Stage 5A score archive hash mismatch")
    if sha256_file(artifact_path) != manifest.get("score_artifact_sha256"):
        raise ValueError("Stage 5A score artifact hash mismatch")
    if sha256_file(comparison_path) != manifest.get("direction_comparison_sha256"):
        raise ValueError("direction comparison hash mismatch")
    if sha256_file(correlation_path) != manifest.get("score_correlation_sha256"):
        raise ValueError("score correlation hash mismatch")
    if sha256_file(Path(source_config["index_path"])) != artifact.get(
        "source_index_sha256"
    ):
        raise ValueError("source index changed after score freeze")
    if sha256_file(Path(entry["target_index"])) != artifact.get(
        "target_index_sha256"
    ):
        raise ValueError("target index changed after score freeze")

    archive = np.load(score_path)
    forbidden = FORBIDDEN_SCORE_FIELDS & set(archive.files)
    if forbidden:
        raise ValueError(f"Stage 5A score archive leaked labels: {sorted(forbidden)}")
    required = {
        "dataset",
        "subject_id",
        "record_id",
        "window_start",
        "analysis_scope",
        *SCORE_FIELDS,
    }
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"Stage 5A archive missing fields: {sorted(missing)}")
    mapping = _label_map(config, entry, source_config)
    labels = _join_labels(archive, mapping)
    if artifact.get("diagnostic_max_batches_per_split") is None:
        if len(labels) != len(mapping):
            raise ValueError(
                f"formal Stage 5A archive covers {len(labels)} of "
                f"{len(mapping)} required windows"
            )

    scopes = archive["analysis_scope"]
    metrics_rows: list[dict] = []
    nested_metrics: dict[str, dict[str, dict]] = {}
    diagnostic = artifact.get("diagnostic_max_batches_per_split") is not None
    for scope in SCOPES:
        mask = scopes == scope
        if not mask.any():
            raise ValueError(f"frozen archive contains no {scope} rows")
        evaluation_dataset = (
            entry["target_dataset"]
            if scope == "target_evaluation"
            else source_name
        )
        nested_metrics[scope] = {}
        for score_type in SCORE_FIELDS:
            metrics = compute_ranking_metrics(
                labels[mask],
                archive[score_type][mask],
                allow_single_class=diagnostic,
            )
            nested_metrics[scope][score_type] = metrics
            metrics_rows.append(
                {
                    "source_dataset": source_name,
                    "evaluation_dataset": evaluation_dataset,
                    "analysis_scope": scope,
                    "score_type": score_type,
                    **metrics,
                }
            )
    metrics_path = output_dir / "split_metrics.csv"
    _write_metrics_csv(metrics_path, metrics_rows)
    _plot_results(
        output_dir,
        archive,
        labels,
        correlation_payload["correlations"],
    )
    result = {
        "source_dataset": source_name,
        "target_dataset": entry["target_dataset"],
        "direction_comparison": comparison,
        "score_correlations": correlation_payload["correlations"],
        "equivalence": correlation_payload["equivalence"],
        "ranking_metrics": nested_metrics,
        "target_label_usage": "post-hoc analysis only",
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_target_evaluation_labels_accessed": True,
    }
    result_path = output_dir / "analysis_result.json"
    _write_json(result_path, result)
    evaluation_manifest = {
        "fit_git": manifest["git"],
        "analysis_git": git_identity(),
        "score_sha256": artifact["score_sha256"],
        "score_artifact_sha256": sha256_file(artifact_path),
        "split_metrics_sha256": sha256_file(metrics_path),
        "analysis_result_sha256": sha256_file(result_path),
        "target_label_usage": "post-hoc analysis only",
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_target_evaluation_labels_accessed": True,
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return result
