"""Stage 5C strong source-only threshold baselines."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.data.ecg_dataset import load_window_rows
from src.evaluation.metrics import compute_binary_metrics
from src.training.reproducibility import git_identity, sha256_file

SCORE_FIELDS = ("classifier_logit_difference", "prototype_score")
FORBIDDEN_SCORE_FIELDS = {"label", "labels", "binary_label", "rhythm_label"}
BASELINES = ("H0", "H1", "P0", "P1")


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


def _write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_protocol(config: dict) -> None:
    if config.get("role") != "source_threshold_baseline":
        raise ValueError("Stage 5C requires role='source_threshold_baseline'")
    if config.get("threshold_selection_label_scope") != "source_validation_only":
        raise ValueError("Stage 5C threshold selection must be source-validation only")
    if config.get("adaptation_time_target_label_access") != "prohibited":
        raise ValueError("Stage 5C must prohibit adaptation-time target labels")
    if sha256_file(Path(config["protocol"])) != config.get("protocol_sha256"):
        raise ValueError("Stage 5C protocol hash mismatch")


def threshold_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    fixed_threshold: float,
) -> tuple[list[dict], dict]:
    """Enumerate realizable thresholds and apply the frozen deterministic rule."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("threshold inputs must be aligned vectors")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("threshold selection requires both binary classes")
    if not np.isfinite(scores).all() or not np.isfinite(fixed_threshold):
        raise ValueError("threshold inputs must be finite")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    tp = fp = 0
    rows = [
        {
            "threshold": float("inf"),
            "tp": 0,
            "fp": 0,
            "tn": negatives,
            "fn": positives,
            "balanced_accuracy": 0.5,
            "macro_f1": negatives / (2 * negatives + positives),
        }
    ]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        tp += int(group.sum())
        fp += int((group == 0).sum())
        fn = positives - tp
        tn = negatives - fp
        sensitivity = tp / positives
        specificity = tn / negatives
        positive_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        negative_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        rows.append(
            {
                "threshold": float(sorted_scores[start]),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "macro_f1": (positive_f1 + negative_f1) / 2,
            }
        )
        start = end

    best_bacc = max(row["balanced_accuracy"] for row in rows)
    candidates = [
        row
        for row in rows
        if np.isclose(row["balanced_accuracy"], best_bacc, atol=1e-12, rtol=0)
    ]
    best_macro_f1 = max(row["macro_f1"] for row in candidates)
    candidates = [
        row
        for row in candidates
        if np.isclose(row["macro_f1"], best_macro_f1, atol=1e-12, rtol=0)
    ]
    best = min(
        candidates,
        key=lambda row: (abs(row["threshold"] - fixed_threshold), row["threshold"]),
    )
    return rows, {
        **best,
        "selection_rule": "argmax_balanced_accuracy",
        "tie_break_rule": "max_macro_f1_then_closest_fixed_then_lower_threshold",
        "candidate_count": len(rows),
        "optimal_balanced_accuracy_tie_count": int(
            sum(
                np.isclose(row["balanced_accuracy"], best_bacc, atol=1e-12, rtol=0)
                for row in rows
            )
        ),
    }


def _validate_score_inputs(config: dict, source_name: str) -> tuple[dict, dict, Path]:
    validate_protocol(config)
    if source_name not in config["sources"]:
        raise ValueError(f"unknown Stage 5C source {source_name}")
    entry = deepcopy(config["sources"][source_name])
    score_dir = Path(entry["stage5a_score_dir"])
    artifact_path = score_dir / "score_artifact.json"
    artifact = _load_json(artifact_path)
    score_manifest = _load_json(score_dir / "run_manifest.json")
    score_path = score_dir / "scores.npz"
    if (
        not artifact.get("frozen")
        or artifact.get("target_labels_accessed") is not False
    ):
        raise ValueError("Stage 5C requires frozen label-free Stage 5A scores")
    if sha256_file(score_path) != artifact.get("score_sha256"):
        raise ValueError("Stage 5A score archive hash mismatch")
    if score_manifest.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5A score manifest does not prove label-free extraction")
    if (
        score_manifest.get("config", {}).get("experiment")
        != "stage5a_head_vs_direction"
    ):
        raise ValueError("Stage 5C input is not a Stage 5A score manifest")
    if artifact.get("diagnostic_max_batches_per_split") is not None:
        raise ValueError("Stage 5C rejects diagnostic Stage 5A scores")
    if sha256_file(artifact_path) != score_manifest.get("score_artifact_sha256"):
        raise ValueError("Stage 5A score artifact hash mismatch")
    for key, path_key in (
        ("checkpoint_sha256", "checkpoint"),
        ("direction_sha256", "direction"),
    ):
        if sha256_file(Path(entry[path_key])) != artifact.get(key):
            raise ValueError(f"Stage 5C {key} mismatch")
    if artifact.get("source_dataset") != source_name:
        raise ValueError("Stage 5C source dataset mismatch")
    if artifact.get("target_dataset") != entry["target_dataset"]:
        raise ValueError("Stage 5C target dataset mismatch")
    return entry, artifact, score_path


def _open_scores(score_path: Path) -> np.lib.npyio.NpzFile:
    archive = np.load(score_path)
    forbidden = FORBIDDEN_SCORE_FIELDS & set(archive.files)
    if forbidden:
        raise ValueError(f"Stage 5C score archive leaked labels: {sorted(forbidden)}")
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
        raise ValueError(f"Stage 5C score archive missing fields: {sorted(missing)}")
    return archive


def _label_map(index_path: Path, *, split_field: str, split_value: str) -> dict:
    kwargs = {split_field: split_value}
    rows = load_window_rows([index_path], **kwargs)
    mapping = {}
    for row in rows:
        key = (row.dataset, row.subject_id, row.record_id, row.start_sample)
        if key in mapping:
            raise ValueError(f"duplicate Stage 5C label key {key}")
        mapping[key] = row.binary_label
    return mapping


def _scope_indices_and_labels(
    archive: np.lib.npyio.NpzFile, scope: str, mapping: dict
) -> tuple[np.ndarray, np.ndarray]:
    scopes = archive["analysis_scope"]
    datasets = archive["dataset"]
    subjects = archive["subject_id"]
    records = archive["record_id"]
    starts = archive["window_start"]
    indices = np.flatnonzero(scopes == scope)
    labels = []
    seen = set()
    for index in indices:
        key = (
            str(datasets[index]),
            str(subjects[index]),
            str(records[index]),
            int(starts[index]),
        )
        if key in seen or key not in mapping:
            raise ValueError(f"invalid or duplicate Stage 5C score key {key}")
        seen.add(key)
        labels.append(mapping[key])
    if len(seen) != len(mapping):
        raise ValueError(
            f"scope {scope} does not exactly cover its current index split"
        )
    return indices, np.asarray(labels, dtype=np.int64)


def _metric_row(
    source: str,
    evaluation_dataset: str,
    scope: str,
    baseline: str,
    score_type: str,
    threshold: float,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    metrics = compute_binary_metrics(labels, scores, threshold=threshold)
    return {
        "source_dataset": source,
        "evaluation_dataset": evaluation_dataset,
        "scope": scope,
        "baseline": baseline,
        "score_type": score_type,
        **metrics,
    }


METRIC_FIELDS = [
    "source_dataset",
    "evaluation_dataset",
    "scope",
    "baseline",
    "score_type",
    "threshold",
    "support",
    "positive_count",
    "negative_count",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "sensitivity",
    "specificity",
    "precision",
    "auroc",
    "auprc",
    "confusion_matrix",
]


def select_source_thresholds(
    config: dict, *, source_name: str, output_override: Path | None = None
) -> dict:
    """Select H1/P1 using source-validation labels without opening target index."""
    config = deepcopy(config)
    entry, artifact, score_path = _validate_score_inputs(config, source_name)
    output_root = output_override or Path(config["output_dir"])
    output_dir = output_root / source_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 5C source output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config = _load_json(Path(entry["source_config"]))
    source_index = Path(source_config["index_path"])
    if sha256_file(source_index) != artifact.get("source_index_sha256"):
        raise ValueError("Stage 5C source index hash mismatch")
    archive = _open_scores(score_path)
    mapping = _label_map(
        source_index, split_field="source_split", split_value="validation"
    )
    indices, labels = _scope_indices_and_labels(archive, "source_validation", mapping)
    direction = _load_json(Path(entry["direction"]))
    fixed = {
        "classifier_logit_difference": 0.0,
        "prototype_score": float(direction["source_fixed_threshold"]),
    }
    baseline_ids = {
        "classifier_logit_difference": ("H0", "H1"),
        "prototype_score": ("P0", "P1"),
    }
    curve_rows = []
    thresholds = {}
    validation_rows = []
    for score_type in SCORE_FIELDS:
        scores = archive[score_type][indices].astype(np.float64)
        curve, selected = threshold_curve(
            labels, scores, fixed_threshold=fixed[score_type]
        )
        fixed_id, optimized_id = baseline_ids[score_type]
        thresholds[fixed_id] = {
            "score_type": score_type,
            "threshold": fixed[score_type],
            "threshold_source": "fixed_source_definition",
        }
        thresholds[optimized_id] = {
            "score_type": score_type,
            "threshold": selected["threshold"],
            "threshold_source": "source_validation_argmax_balanced_accuracy",
            "selection": selected,
        }
        for row in curve:
            curve_rows.append(
                {"source_dataset": source_name, "score_type": score_type, **row}
            )
        validation_rows.extend(
            [
                _metric_row(
                    source_name,
                    source_name,
                    "source_validation",
                    fixed_id,
                    score_type,
                    fixed[score_type],
                    labels,
                    scores,
                ),
                _metric_row(
                    source_name,
                    source_name,
                    "source_validation",
                    optimized_id,
                    score_type,
                    selected["threshold"],
                    labels,
                    scores,
                ),
            ]
        )
    curve_path = output_dir / "source_validation_curves.csv"
    validation_path = output_dir / "source_validation_results.csv"
    _write_rows(
        curve_path,
        curve_rows,
        [
            "source_dataset",
            "score_type",
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    _write_rows(validation_path, validation_rows, METRIC_FIELDS)
    threshold_path = output_dir / "threshold_artifact.json"
    threshold_artifact = {
        "frozen": True,
        "source_dataset": source_name,
        "target_dataset": entry["target_dataset"],
        "selection_label_scope": "source_validation_only",
        "target_labels_accessed": False,
        "adaptation_time_target_label_access": "prohibited",
        "score_sha256": artifact["score_sha256"],
        "source_index_sha256": artifact["source_index_sha256"],
        "target_index_sha256_from_frozen_score_artifact": artifact[
            "target_index_sha256"
        ],
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "direction_sha256": artifact["direction_sha256"],
        "thresholds": thresholds,
        "source_validation_count": len(labels),
        "source_validation_curves_sha256": sha256_file(curve_path),
        "source_validation_results_sha256": sha256_file(validation_path),
    }
    _write_json(threshold_path, threshold_artifact)
    selection_manifest = {
        "git": git_identity(),
        "config": config,
        "source_dataset": source_name,
        "target_index_opened": False,
        "target_labels_accessed": False,
        "threshold_artifact_sha256": sha256_file(threshold_path),
    }
    _write_json(output_dir / "selection_manifest.json", selection_manifest)
    return {
        "source_dataset": source_name,
        "target_dataset": entry["target_dataset"],
        "target_labels_accessed": False,
        "thresholds": {key: value["threshold"] for key, value in thresholds.items()},
        "output_dir": str(output_dir),
    }


def evaluate_source_thresholds(
    config: dict, *, source_name: str, output_override: Path | None = None
) -> dict:
    """Evaluate already-frozen thresholds; target labels enter only here."""
    config = deepcopy(config)
    entry, score_artifact, score_path = _validate_score_inputs(config, source_name)
    output_root = output_override or Path(config["output_dir"])
    output_dir = output_root / source_name
    threshold_path = output_dir / "threshold_artifact.json"
    selection_path = output_dir / "selection_manifest.json"
    threshold_artifact = _load_json(threshold_path)
    selection_manifest = _load_json(selection_path)
    if not threshold_artifact.get("frozen"):
        raise ValueError("Stage 5C thresholds are not frozen")
    if threshold_artifact.get("target_labels_accessed") is not False:
        raise ValueError("Stage 5C threshold selection accessed target labels")
    if selection_manifest.get("target_index_opened") is not False:
        raise ValueError("Stage 5C selection did not prove target-index isolation")
    if sha256_file(threshold_path) != selection_manifest.get(
        "threshold_artifact_sha256"
    ):
        raise ValueError("Stage 5C threshold artifact hash mismatch")
    if threshold_artifact.get("score_sha256") != score_artifact["score_sha256"]:
        raise ValueError("Stage 5C threshold score provenance mismatch")
    if sha256_file(
        output_dir / "source_validation_curves.csv"
    ) != threshold_artifact.get("source_validation_curves_sha256"):
        raise ValueError("Stage 5C source-validation curve hash mismatch")

    source_config = _load_json(Path(entry["source_config"]))
    source_index = Path(source_config["index_path"])
    target_index = Path(entry["target_index"])
    if sha256_file(source_index) != threshold_artifact["source_index_sha256"]:
        raise ValueError("Stage 5C source index changed after threshold freeze")
    if (
        sha256_file(target_index)
        != threshold_artifact["target_index_sha256_from_frozen_score_artifact"]
    ):
        raise ValueError("Stage 5C target index changed after threshold freeze")
    archive = _open_scores(score_path)
    scopes = {
        "source_test": (
            source_name,
            _label_map(source_index, split_field="source_split", split_value="test"),
        ),
        "target_evaluation": (
            entry["target_dataset"],
            _label_map(
                target_index, split_field="target_split", split_value="evaluation"
            ),
        ),
    }
    rows_by_scope: dict[str, list[dict]] = {}
    nested: dict[str, dict] = {}
    for scope, (evaluation_dataset, mapping) in scopes.items():
        indices, labels = _scope_indices_and_labels(archive, scope, mapping)
        rows = []
        nested[scope] = {}
        for baseline in BASELINES:
            threshold_entry = threshold_artifact["thresholds"][baseline]
            score_type = threshold_entry["score_type"]
            row = _metric_row(
                source_name,
                evaluation_dataset,
                scope,
                baseline,
                score_type,
                float(threshold_entry["threshold"]),
                labels,
                archive[score_type][indices].astype(np.float64),
            )
            rows.append(row)
            nested[scope][baseline] = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "source_dataset",
                    "evaluation_dataset",
                    "scope",
                    "baseline",
                    "score_type",
                }
            }
        for fixed_id, optimized_id in (("H0", "H1"), ("P0", "P1")):
            if nested[scope][fixed_id]["auroc"] != nested[scope][optimized_id]["auroc"]:
                raise RuntimeError("threshold-only baseline changed AUROC")
            if nested[scope][fixed_id]["auprc"] != nested[scope][optimized_id]["auprc"]:
                raise RuntimeError("threshold-only baseline changed AUPRC")
        rows_by_scope[scope] = rows

    source_test_path = output_dir / "source_test_results.csv"
    target_path = output_dir / "target_results.csv"
    _write_rows(source_test_path, rows_by_scope["source_test"], METRIC_FIELDS)
    _write_rows(target_path, rows_by_scope["target_evaluation"], METRIC_FIELDS)
    result = {
        "source_dataset": source_name,
        "target_dataset": entry["target_dataset"],
        "threshold_selection_target_labels_accessed": False,
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_target_evaluation_labels_accessed": True,
        "thresholds": threshold_artifact["thresholds"],
        "metrics": nested,
    }
    result_path = output_dir / "analysis_result.json"
    _write_json(result_path, result)
    evaluation_manifest = {
        "selection_git": selection_manifest["git"],
        "evaluation_git": git_identity(),
        "threshold_artifact_sha256": sha256_file(threshold_path),
        "score_sha256": score_artifact["score_sha256"],
        "source_test_results_sha256": sha256_file(source_test_path),
        "target_results_sha256": sha256_file(target_path),
        "analysis_result_sha256": sha256_file(result_path),
        "threshold_selection_target_labels_accessed": False,
        "adaptation_time_target_labels_accessed": False,
        "post_freeze_target_evaluation_labels_accessed": True,
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return result


def summarize_source_thresholds(
    config: dict, *, output_override: Path | None = None
) -> dict:
    """Combine the two completed development-transfer Stage 5C analyses."""
    config = deepcopy(config)
    validate_protocol(config)
    output_root = output_override or Path(config["output_dir"])
    threshold_rows = []
    curve_rows = []
    target_rows = []
    results = {}
    for source_name, entry in config["sources"].items():
        source_dir = output_root / source_name
        result = _load_json(source_dir / "analysis_result.json")
        manifest = _load_json(source_dir / "evaluation_manifest.json")
        if sha256_file(source_dir / "analysis_result.json") != manifest.get(
            "analysis_result_sha256"
        ):
            raise ValueError(f"Stage 5C result hash mismatch for {source_name}")
        results[source_name] = result
        for baseline, threshold in result["thresholds"].items():
            threshold_rows.append(
                {
                    "source_dataset": source_name,
                    "target_dataset": entry["target_dataset"],
                    "baseline": baseline,
                    "score_type": threshold["score_type"],
                    "threshold": threshold["threshold"],
                    "threshold_source": threshold["threshold_source"],
                }
            )
        with (source_dir / "source_validation_curves.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            curve_rows.extend(csv.DictReader(handle))
        with (source_dir / "target_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            target_rows.extend(csv.DictReader(handle))
    thresholds_path = output_root / "thresholds.csv"
    curves_path = output_root / "source_validation_curves.csv"
    targets_path = output_root / "target_results.csv"
    _write_rows(
        thresholds_path,
        threshold_rows,
        [
            "source_dataset",
            "target_dataset",
            "baseline",
            "score_type",
            "threshold",
            "threshold_source",
        ],
    )
    _write_rows(
        curves_path,
        curve_rows,
        [
            "source_dataset",
            "score_type",
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    _write_rows(targets_path, target_rows, METRIC_FIELDS)
    summary = {
        "experiment": config["experiment"],
        "development_transfers_only": True,
        "final_transfer_labels_accessed": False,
        "threshold_selection_label_scope": "source_validation_only",
        "adaptation_time_target_label_access": "prohibited",
        "sources": results,
        "artifacts": {
            "thresholds_sha256": sha256_file(thresholds_path),
            "source_validation_curves_sha256": sha256_file(curves_path),
            "target_results_sha256": sha256_file(targets_path),
        },
    }
    _write_json(output_root / "analysis_result.json", summary)
    _write_json(
        output_root / "run_manifest.json",
        {"git": git_identity(), "config": config, "artifacts": summary["artifacts"]},
    )
    return summary
