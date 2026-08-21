"""Frozen Main M3 evaluation of the Main M2 model on three target datasets."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    ECGWindowDataset,
    load_unlabeled_target_rows,
    load_window_rows,
)
from src.evaluation.metrics import compute_binary_metrics
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


FORBIDDEN_SCORE_FIELDS = {"label", "labels", "binary_label", "rhythm_label"}
IDENTITY_FIELDS = ("dataset", "subject_id", "record_id", "window_start")
SCORE_FIELD = "raw_logit_difference"
PRIMARY_METRICS = ("auroc", "auprc")
OPERATING_METRICS = ("balanced_accuracy", "macro_f1", "mcc")
ALL_METRICS = PRIMARY_METRICS + OPERATING_METRICS


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Main M3 CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_score_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    forbidden = FORBIDDEN_SCORE_FIELDS & set(arrays)
    if forbidden:
        raise ValueError(f"target score archive cannot contain labels: {sorted(forbidden)}")
    required = set(IDENTITY_FIELDS) | {SCORE_FIELD}
    if set(arrays) != required:
        raise ValueError("target score archive schema changed")
    lengths = {len(np.asarray(value)) for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("target score archive arrays are empty or misaligned")
    scores = np.asarray(arrays[SCORE_FIELD])
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("target scores contain NaN/Inf or are not one-dimensional")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def identities_from_archive(archive: np.lib.npyio.NpzFile) -> list[tuple]:
    return list(
        zip(
            map(str, archive["dataset"]),
            map(str, archive["subject_id"]),
            map(str, archive["record_id"]),
            map(int, archive["window_start"]),
        )
    )


def identities_from_rows(rows) -> list[tuple]:
    return [
        (row.dataset, row.subject_id, row.record_id, row.start_sample) for row in rows
    ]


def validate_config(config: dict) -> tuple[dict, dict, dict]:
    if config.get("role") != "frozen_main_m2_three_target_evaluation":
        raise ValueError("Main M3 config role changed")
    if list(config.get("targets", {})) != ["cpsc2021", "ltaf_clean1h", "shdb"]:
        raise ValueError("Main M3 target order or cohort changed")
    if config.get("score_definition") != "logit_AF_minus_logit_nonAF":
        raise ValueError("Main M3 score definition changed")
    if config.get("threshold_policy") != "single_AFDB_OOF_raw_logit_threshold":
        raise ValueError("Main M3 threshold policy changed")
    for key, hash_key in (
        ("governing_plan", "governing_plan_sha256"),
        ("m2_protocol", "m2_protocol_sha256"),
        ("m2_selection", "m2_selection_sha256"),
        ("m2_epoch_rule", "m2_epoch_rule_sha256"),
        ("m2_checkpoint", "m2_checkpoint_sha256"),
        ("m2_result", "m2_result_sha256"),
        ("reference_result", "reference_result_sha256"),
    ):
        if sha256_file(Path(config[key])) != config[hash_key]:
            raise ValueError(f"Main M3 frozen input changed: {key}")
    for target, entry in config["targets"].items():
        if sha256_file(Path(entry["index_path"])) != entry["index_sha256"]:
            raise ValueError(f"Main M3 target index changed: {target}")

    selection = load_json(Path(config["m2_selection"]))
    epoch_rule = load_json(Path(config["m2_epoch_rule"]))
    result = load_json(Path(config["m2_result"]))
    if not selection.get("frozen") or selection.get("target_data_accessed") is not False:
        raise ValueError("Main M2 selection is not frozen source-only")
    if not epoch_rule.get("frozen") or epoch_rule.get("target_data_accessed") is not False:
        raise ValueError("Main M2 epoch rule is not frozen source-only")
    if result.get("target_data_accessed") is not False or not result.get("formal"):
        raise ValueError("Main M2 final result is not formal source-only")
    threshold = float(config["source_oof_threshold"])
    if not np.isclose(threshold, float(selection["selected_threshold"]), atol=0, rtol=0):
        raise ValueError("Main M3 threshold differs from frozen M2 OOF threshold")
    if not np.isclose(
        float(selection["selected_lambda"]), float(result["lambda_axis"]), atol=0, rtol=0
    ):
        raise ValueError("Main M2 selected and final lambdas differ")
    if int(epoch_rule["final_epoch"]) != int(result["epochs_completed"]):
        raise ValueError("Main M2 final epoch provenance differs")
    if result["checkpoint_sha256"] != config["m2_checkpoint_sha256"]:
        raise ValueError("Main M2 result checkpoint hash differs")
    return selection, epoch_rule, result


def load_frozen_model(config: dict, device: torch.device) -> SourceMedTSTTT:
    selection, epoch_rule, result = validate_config(config)
    checkpoint = torch.load(
        Path(config["m2_checkpoint"]), map_location=device, weights_only=False
    )
    provenance = checkpoint.get("provenance", {})
    if not provenance.get("formal") or provenance.get("target_data_accessed") is not False:
        raise ValueError("Main M2 checkpoint is not formal source-only")
    expected = {
        "lambda_axis": float(selection["selected_lambda"]),
        "epochs": int(epoch_rule["final_epoch"]),
        "selected_oof_threshold": float(selection["selected_threshold"]),
    }
    for key, value in expected.items():
        actual = provenance.get(key)
        if isinstance(value, float):
            valid = np.isclose(float(actual), value, atol=0, rtol=0)
        else:
            valid = int(actual) == value
        if not valid:
            raise ValueError(f"Main M2 checkpoint provenance mismatch: {key}")
    model_config = provenance.get("config", {}).get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Main M2 checkpoint lacks model config")
    model = SourceMedTSTTT(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if int(checkpoint.get("epoch", -1)) != int(result["epochs_completed"]):
        raise ValueError("Main M2 checkpoint epoch differs from final result")
    return model


def score_target(
    config: dict,
    *,
    target: str,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches: int | None = None,
) -> dict:
    validate_config(config)
    if target not in config["targets"]:
        raise ValueError("unknown Main M3 target")
    diagnostic = max_batches is not None
    if diagnostic and output_override is None:
        raise ValueError("Main M3 diagnostics require an explicit output override")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    git = git_identity()
    if not diagnostic and git.get("dirty"):
        raise ValueError("formal Main M3 scoring requires a clean Git worktree")

    entry = config["targets"][target]
    rows = load_unlabeled_target_rows(
        [Path(entry["index_path"])], target_split="evaluation"
    )
    if not diagnostic and len(rows) != int(entry["expected_evaluation_windows"]):
        raise ValueError("Main M3 target evaluation count changed")
    if diagnostic:
        rows = rows[: max_batches * int(config["extraction"]["batch_size"])]
    expected_ids = identities_from_rows(rows)
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Main M3 target index contains duplicate identities")

    root = Path(output_override or config["output_dir"])
    output = root / target
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Main M3 target output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    model = load_frozen_model(config, device)
    loader = DataLoader(
        ECGWindowDataset(rows, data_root=Path(entry["data_root"]), expose_label=False),
        batch_size=int(config["extraction"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["extraction"]["num_workers"]),
        pin_memory=False,
    )
    parts: dict[str, list] = {field: [] for field in IDENTITY_FIELDS}
    parts[SCORE_FIELD] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if not bool((batch["y"] == -1).all()):
                raise ValueError("Main M3 scoring exposed target labels")
            logits = model(batch["x"].to(device))
            scores = logits[:, 1] - logits[:, 0]
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Main M3 scoring produced NaN/Inf")
            metadata = batch["metadata"]
            parts["dataset"].extend(map(str, metadata["dataset"]))
            parts["subject_id"].extend(map(str, metadata["subject_id"]))
            parts["record_id"].extend(map(str, metadata["record_id"]))
            parts["window_start"].extend(map(int, metadata["window_start"].tolist()))
            parts[SCORE_FIELD].extend(scores.cpu().numpy().astype(np.float32).tolist())
            if batch_index % int(config["extraction"]["progress_every_batches"]) == 0:
                print(
                    f"phase=main_m3_score target={target} batch={batch_index} "
                    f"samples={len(parts[SCORE_FIELD])} "
                    f"seconds={time.perf_counter()-started:.1f}",
                    flush=True,
                )
    arrays = {
        "dataset": np.asarray(parts["dataset"], dtype=np.str_),
        "subject_id": np.asarray(parts["subject_id"], dtype=np.str_),
        "record_id": np.asarray(parts["record_id"], dtype=np.str_),
        "window_start": np.asarray(parts["window_start"], dtype=np.int64),
        SCORE_FIELD: np.asarray(parts[SCORE_FIELD], dtype=np.float32),
    }
    actual_ids = list(
        zip(
            map(str, arrays["dataset"]),
            map(str, arrays["subject_id"]),
            map(str, arrays["record_id"]),
            map(int, arrays["window_start"]),
        )
    )
    if actual_ids != expected_ids:
        raise ValueError("Main M3 scored identity order/coverage changed")
    score_path = output / "scores.npz"
    write_score_archive(score_path, arrays)
    artifact = {
        "frozen": True,
        "formal": not diagnostic,
        "target": target,
        "target_labels_accessed": False,
        "target_label_fields_parsed": False,
        "diagnostic_max_batches": max_batches,
        "windows": len(actual_ids),
        "score_definition": config["score_definition"],
        "source_oof_threshold": float(config["source_oof_threshold"]),
        "target_specific_tuning": False,
        "target_index_sha256": entry["index_sha256"],
        "m2_selection_sha256": config["m2_selection_sha256"],
        "m2_checkpoint_sha256": config["m2_checkpoint_sha256"],
        "score_sha256": sha256_file(score_path),
    }
    artifact_path = output / "score_artifact.json"
    write_json(artifact_path, artifact)
    write_json(
        output / "extraction_manifest.json",
        {
            "git": git,
            "environment": environment_snapshot(device),
            "formal": not diagnostic,
            "target_labels_accessed": False,
            "score_artifact_sha256": sha256_file(artifact_path),
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    return artifact


def audit_score_archive(config: dict, target: str, *, require_formal: bool = True) -> dict:
    validate_config(config)
    if target not in config["targets"]:
        raise ValueError("unknown Main M3 target")
    output = Path(config["output_dir"]) / target
    artifact_path = output / "score_artifact.json"
    manifest_path = output / "extraction_manifest.json"
    score_path = output / "scores.npz"
    artifact = load_json(artifact_path)
    manifest = load_json(manifest_path)
    if not artifact.get("frozen") or artifact.get("target_labels_accessed") is not False:
        raise ValueError("Main M3 target score is not frozen label-free")
    if artifact.get("target_label_fields_parsed") is not False:
        raise ValueError("Main M3 scoring parsed target-label fields")
    if manifest.get("target_labels_accessed") is not False:
        raise ValueError("Main M3 extraction manifest is not label-free")
    if require_formal and (
        not artifact.get("formal")
        or not manifest.get("formal")
        or artifact.get("diagnostic_max_batches") is not None
    ):
        raise ValueError("Main M3 formal evaluation rejects diagnostic scores")
    if sha256_file(score_path) != artifact.get("score_sha256"):
        raise ValueError("Main M3 target score hash mismatch")
    if sha256_file(artifact_path) != manifest.get("score_artifact_sha256"):
        raise ValueError("Main M3 target score artifact hash mismatch")
    entry = config["targets"][target]
    if artifact.get("target_index_sha256") != entry["index_sha256"]:
        raise ValueError("Main M3 target index provenance mismatch")
    if artifact.get("m2_checkpoint_sha256") != config["m2_checkpoint_sha256"]:
        raise ValueError("Main M3 checkpoint provenance mismatch")
    if not np.isclose(
        float(artifact.get("source_oof_threshold")),
        float(config["source_oof_threshold"]),
        atol=0,
        rtol=0,
    ):
        raise ValueError("Main M3 target threshold provenance mismatch")
    archive = np.load(score_path)
    if set(archive.files) != set(IDENTITY_FIELDS) | {SCORE_FIELD}:
        raise ValueError("Main M3 target score archive schema changed")
    if FORBIDDEN_SCORE_FIELDS & set(archive.files):
        raise ValueError("Main M3 target score archive leaked labels")
    scores = archive[SCORE_FIELD]
    ids = identities_from_archive(archive)
    if len(ids) != len(set(ids)) or len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Main M3 target score archive is invalid")
    expected_rows = load_unlabeled_target_rows(
        [Path(entry["index_path"])], target_split="evaluation"
    )
    expected_ids = identities_from_rows(expected_rows)
    if ids != expected_ids or len(ids) != int(entry["expected_evaluation_windows"]):
        raise ValueError("Main M3 formal score archive does not exactly cover target")
    return {
        "target": target,
        "windows": len(ids),
        "score_sha256": artifact["score_sha256"],
        "target_labels_accessed": False,
        "coverage_exact": True,
        "finite": True,
        "schema_label_free": True,
    }


def mechanism_statistics(labels: np.ndarray, scores: np.ndarray, *, bins: int) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("mechanism inputs must be aligned one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("mechanism statistics require both target classes")
    if bins <= 1 or not np.isfinite(scores).all():
        raise ValueError("invalid mechanism score inputs")
    nonaf = scores[labels == 0]
    af = scores[labels == 1]
    mean_nonaf, mean_af = float(nonaf.mean()), float(af.mean())
    std_nonaf, std_af = float(nonaf.std()), float(af.std())
    pooled = float(np.sqrt((std_nonaf**2 + std_af**2) / 2))
    if pooled <= 1e-12:
        raise ValueError("target class distributions have zero pooled variance")
    low, high = float(scores.min()), float(scores.max())
    if not high > low:
        raise ValueError("target score range is degenerate")
    h0, _ = np.histogram(nonaf, bins=bins, range=(low, high))
    h1, _ = np.histogram(af, bins=bins, range=(low, high))
    overlap = float(np.minimum(h0 / h0.sum(), h1 / h1.sum()).sum())
    gap = mean_af - mean_nonaf
    return {
        "score_range_rule": "target_unlabeled_observed_min_max",
        "score_min": low,
        "score_max": high,
        "histogram_bins": int(bins),
        "mu_nonaf": mean_nonaf,
        "mu_af": mean_af,
        "std_nonaf": std_nonaf,
        "std_af": std_af,
        "class_gap": gap,
        "pooled_std": pooled,
        "d_prime": gap / pooled,
        "histogram_overlap_coefficient": overlap,
    }


def decision_status(current: dict, reference: dict) -> dict:
    deltas = {metric: float(current[metric] - reference[metric]) for metric in ALL_METRICS}
    ranking_up = all(deltas[metric] > 0 for metric in PRIMARY_METRICS)
    operating_up = sum(deltas[metric] > 0 for metric in OPERATING_METRICS)
    if ranking_up and operating_up >= 2:
        status = "strong_success_pending_target_consistency"
    elif ranking_up:
        status = "partial_success"
    elif all(deltas[metric] < 0 for metric in PRIMARY_METRICS) and sum(
        deltas[metric] < 0 for metric in OPERATING_METRICS
    ) >= 2:
        status = "failure"
    else:
        status = "mixed_or_neutral"
    return {"status": status, "deltas": deltas}


def _label_map(index_path: Path) -> dict[tuple, int]:
    rows = load_window_rows([index_path], target_split="evaluation")
    mapping = {}
    for row in rows:
        key = (row.dataset, row.subject_id, row.record_id, row.start_sample)
        if key in mapping:
            raise ValueError("duplicate target label identity")
        mapping[key] = row.binary_label
    return mapping


def _reference_rows(config: dict) -> dict[str, dict]:
    reference = load_json(Path(config["reference_result"]))
    if reference.get("selected_alpha") != 0.0:
        raise ValueError("frozen reference is not the R2/M1 head endpoint")
    rows = {}
    for target in config["targets"]:
        selected = [row for row in reference["per_target"][target] if row["selected"]]
        if len(selected) != 1:
            raise ValueError("frozen reference target row is ambiguous")
        rows[target] = selected[0]
    return rows


def evaluate_targets(config: dict) -> dict:
    selection, _, m2_result = validate_config(config)
    root = Path(config["output_dir"])
    if (root / "analysis_result.json").exists():
        raise FileExistsError("Main M3 analysis result already exists")

    # This global gate completes all label-free audits before any label CSV is read.
    score_audits = {
        target: audit_score_archive(config, target, require_formal=True)
        for target in config["targets"]
    }
    reference_rows = _reference_rows(config)
    per_target = {}
    metric_rows = []
    mechanism_rows = []
    for target, entry in config["targets"].items():
        output = root / target
        artifact = load_json(output / "score_artifact.json")
        archive = np.load(output / "scores.npz")
        ids = identities_from_archive(archive)
        mapping = _label_map(Path(entry["index_path"]))
        if set(ids) != set(mapping) or len(ids) != len(mapping):
            raise ValueError("target labels do not match frozen score coverage")
        labels = np.asarray([mapping[identity] for identity in ids], dtype=np.int64)
        scores = archive[SCORE_FIELD].astype(np.float64)
        metrics = compute_binary_metrics(
            labels, scores, threshold=float(config["source_oof_threshold"])
        )
        mechanism = mechanism_statistics(
            labels, scores, bins=int(config["mechanism"]["histogram_bins"])
        )
        reference = reference_rows[target]
        deltas = {
            metric: float(metrics[metric] - reference[metric]) for metric in ALL_METRICS
        }
        row = {
            "target": target,
            **metrics,
            **{f"reference_{metric}": float(reference[metric]) for metric in ALL_METRICS},
            **{f"delta_{metric}": value for metric, value in deltas.items()},
        }
        mechanism_row = {"target": target, **mechanism}
        metric_rows.append(row)
        mechanism_rows.append(mechanism_row)
        per_target[target] = {
            "metrics": metrics,
            "mechanism_post_freeze": mechanism,
            "reference": {metric: float(reference[metric]) for metric in ALL_METRICS},
            "deltas": deltas,
        }
        metrics_path = output / "metrics.json"
        write_json(
            metrics_path,
            {
                "post_freeze_target_labels_accessed": True,
                "target_specific_tuning": False,
                "score_sha256": artifact["score_sha256"],
                **per_target[target],
            },
        )
        write_json(
            output / "evaluation_manifest.json",
            {
                "analysis_git": git_identity(),
                "post_freeze_target_labels_accessed": True,
                "source_oof_threshold": float(config["source_oof_threshold"]),
                "score_sha256": artifact["score_sha256"],
                "metrics_sha256": sha256_file(metrics_path),
            },
        )

    current_means = {
        metric: float(np.mean([row["metrics"][metric] for row in per_target.values()]))
        for metric in ALL_METRICS
    }
    reference_means = {
        metric: float(np.mean([row["reference"][metric] for row in per_target.values()]))
        for metric in ALL_METRICS
    }
    decision = decision_status(current_means, reference_means)
    consistent_targets = sum(
        row["deltas"]["auroc"] >= 0
        and row["deltas"]["auprc"] >= 0
        and (row["deltas"]["auroc"] > 0 or row["deltas"]["auprc"] > 0)
        for row in per_target.values()
    )
    if decision["status"] == "strong_success_pending_target_consistency":
        decision["status"] = (
            "strong_success" if consistent_targets >= 2 else "partial_success"
        )
    decision["targets_with_consistent_ranking_improvement"] = int(consistent_targets)
    result = {
        "experiment": config["experiment"],
        "formal": True,
        "model": "Ours-Axis",
        "m2_selected_lambda": float(selection["selected_lambda"]),
        "source_oof_threshold": float(config["source_oof_threshold"]),
        "m2_checkpoint_sha256": config["m2_checkpoint_sha256"],
        "target_specific_tuning": False,
        "all_target_scores_frozen_before_any_label_access": True,
        "target_labels_used_only_after_score_freeze": True,
        "score_audits": score_audits,
        "per_target": per_target,
        "three_target_means": current_means,
        "reference_three_target_means": reference_means,
        "decision": decision,
        "final_head_axis_cosine": float(m2_result["final_head_axis_cosine"]),
        "final_head_axis_angle_degrees": float(m2_result["final_head_axis_angle_degrees"]),
    }
    result_path = root / "analysis_result.json"
    metrics_csv = root / "target_metrics.csv"
    mechanism_csv = root / "mechanism_metrics.csv"
    write_json(result_path, result)
    write_csv(metrics_csv, metric_rows)
    write_csv(mechanism_csv, mechanism_rows)
    write_json(
        root / "run_manifest.json",
        {
            "git": git_identity(),
            "formal": True,
            "target_specific_tuning": False,
            "analysis_result_sha256": sha256_file(result_path),
            "target_metrics_sha256": sha256_file(metrics_csv),
            "mechanism_metrics_sha256": sha256_file(mechanism_csv),
        },
    )
    return result


def completion_audit(config: dict) -> dict:
    validate_config(config)
    root = Path(config["output_dir"])
    score_audits = {
        target: audit_score_archive(config, target, require_formal=True)
        for target in config["targets"]
    }
    result_path = root / "analysis_result.json"
    manifest_path = root / "run_manifest.json"
    manifest = load_json(manifest_path)
    result = load_json(result_path)
    if sha256_file(result_path) != manifest.get("analysis_result_sha256"):
        raise ValueError("Main M3 analysis result hash mismatch")
    if not result.get("all_target_scores_frozen_before_any_label_access"):
        raise ValueError("Main M3 global score-freeze invariant missing")
    if result.get("target_specific_tuning") is not False:
        raise ValueError("Main M3 target-specific tuning invariant failed")
    for target in config["targets"]:
        evaluation = load_json(root / target / "evaluation_manifest.json")
        if not evaluation.get("post_freeze_target_labels_accessed"):
            raise ValueError("Main M3 post-freeze label access not recorded")
        if evaluation.get("score_sha256") != score_audits[target]["score_sha256"]:
            raise ValueError("Main M3 evaluated score differs from frozen score")
    audit = {
        "status": "PASS",
        "formal": True,
        "targets": list(config["targets"]),
        "total_windows": int(sum(x["windows"] for x in score_audits.values())),
        "score_audits": score_audits,
        "m2_checkpoint_sha256": config["m2_checkpoint_sha256"],
        "source_oof_threshold": float(config["source_oof_threshold"]),
        "target_specific_tuning": False,
        "target_labels_used_only_after_score_freeze": True,
        "analysis_result_sha256": manifest["analysis_result_sha256"],
    }
    write_json(root / "completion_audit.json", audit)
    return audit
