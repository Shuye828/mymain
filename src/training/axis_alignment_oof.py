"""Finalize and select Main M2 AFDB source-only OOF candidates."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGWindowDataset, load_window_rows
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import threshold_curve
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.afdb_source_protocol import (
    fold_subject_partitions,
    median_best_epoch,
    read_fold_assignments,
)
from src.training.axis_alignment import (
    _load_fold_rows,
    extract_source_axis,
    head_axis_geometry,
    lambda_slug,
    load_json,
    validate_m2_config,
    write_json,
)
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Main M2 CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _identity_arrays(identities: list[tuple]) -> dict[str, np.ndarray]:
    return {
        "dataset": np.asarray([item[0] for item in identities]),
        "subject_id": np.asarray([item[1] for item in identities]),
        "record_id": np.asarray([item[2] for item in identities]),
        "window_start": np.asarray([item[3] for item in identities], dtype=np.int64),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError("Main M2 OOF arrays are misaligned")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


@torch.no_grad()
def _extract_logits(
    model: SourceMedTSTTT,
    rows: list,
    *,
    config: dict,
    device: torch.device,
    phase: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    loader = DataLoader(
        ECGWindowDataset(rows, data_root=Path(config["data_root"])),
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    identities: list[tuple] = []
    started = time.perf_counter()
    progress = int(config["training"]["progress_every_batches"])
    for batch_index, batch in enumerate(loader, 1):
        logits = model(batch["x"].to(device))
        score = logits[:, 1] - logits[:, 0]
        if not torch.isfinite(score).all():
            raise FloatingPointError("Main M2 OOF logits contain NaN/Inf")
        scores.append(score.cpu().numpy().astype(np.float64))
        labels.append(batch["y"].numpy().astype(np.int8))
        metadata = batch["metadata"]
        identities.extend(
            zip(
                map(str, metadata["dataset"]),
                map(str, metadata["subject_id"]),
                map(str, metadata["record_id"]),
                map(int, metadata["window_start"].tolist()),
            )
        )
        if batch_index % progress == 0:
            print(
                f"phase={phase} batch={batch_index} samples={len(identities)} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    return np.concatenate(scores), np.concatenate(labels), identities


def finalize_lambda_oof(
    config: dict,
    *,
    lambda_axis: float,
    device_request: str = "auto",
    output_override: Path | None = None,
) -> dict:
    validate_m2_config(config)
    if float(lambda_axis) not in map(float, config["lambdas"]):
        raise ValueError("unknown Main M2 lambda")
    diagnostic = output_override is not None
    git = git_identity()
    if not diagnostic and git.get("dirty") is not False:
        raise ValueError("formal Main M2 OOF requires a clean Git worktree")
    output = output_override or (
        Path(config["output_dir"]) / "oof" / lambda_slug(lambda_axis)
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Main M2 OOF output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    assignments = read_fold_assignments(Path(config["fold_manifest"]))
    expected_rows = load_window_rows([Path(config["index_path"])])
    expected_ids = [
        (row.dataset, row.subject_id, row.record_id, row.start_sample)
        for row in expected_rows
    ]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("AFDB index contains duplicate identities")

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_folds: list[np.ndarray] = []
    all_ids: list[tuple] = []
    fold_geometry: list[dict] = []
    best_epochs: list[int] = []
    started = time.perf_counter()
    for fold in range(5):
        fold_dir = (
            Path(config["output_dir"])
            / "folds"
            / lambda_slug(lambda_axis)
            / f"fold_{fold}"
        )
        result = load_json(fold_dir / "result.json")
        if not result.get("formal") or result.get("target_data_accessed") is not False:
            raise ValueError(f"Main M2 fold {fold} is not a formal source-only run")
        checkpoint_path = fold_dir / "best.pt"
        if sha256_file(checkpoint_path) != result["best_checkpoint_sha256"]:
            raise ValueError(f"Main M2 fold {fold} checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        provenance = payload.get("provenance", {})
        training_subjects, validation_subjects = fold_subject_partitions(
            assignments, fold
        )
        if (
            provenance.get("fold") != fold
            or float(provenance.get("lambda_axis")) != float(lambda_axis)
            or set(provenance.get("training_subjects", [])) != training_subjects
            or set(provenance.get("validation_subjects", [])) != validation_subjects
            or provenance.get("target_data_accessed") is not False
            or provenance.get("protocol_sha256") != config["protocol_sha256"]
            or provenance.get("index_sha256") != config["index_sha256"]
            or provenance.get("fold_manifest_sha256")
            != config["fold_manifest_sha256"]
            or provenance.get("git", {}).get("dirty") is not False
            or provenance.get("git", {}).get("commit") != git.get("commit")
        ):
            raise ValueError(f"Main M2 fold {fold} provenance mismatch")
        model = SourceMedTSTTT(**config["model"]).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        train_rows, validation_rows, _, _ = _load_fold_rows(config, fold)
        axis_loader = DataLoader(
            ECGWindowDataset(train_rows, data_root=Path(config["data_root"])),
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["training"]["num_workers"]),
        )
        axis, axis_stats = extract_source_axis(
            model,
            axis_loader,
            device=device,
            progress_every_batches=int(config["training"]["progress_every_batches"]),
            phase=f"m2_oof_axis_lambda_{lambda_axis}_fold_{fold}",
        )
        cosine, angle = head_axis_geometry(model, axis)
        if not np.isclose(
            cosine,
            float(result["best_head_axis_cosine"]),
            atol=2e-6,
            rtol=0,
        ) or not np.isclose(
            angle,
            float(result["best_head_axis_angle_degrees"]),
            atol=2e-4,
            rtol=0,
        ):
            raise ValueError(f"Main M2 fold {fold} stored geometry mismatch")
        scores, labels, identities = _extract_logits(
            model,
            validation_rows,
            config=config,
            device=device,
            phase=f"m2_oof_logits_lambda_{lambda_axis}_fold_{fold}",
        )
        if any(identity[1] in training_subjects for identity in identities):
            raise ValueError(f"Main M2 fold {fold} leaks a training subject")
        all_scores.append(scores)
        all_labels.append(labels)
        all_folds.append(np.full(len(labels), fold, dtype=np.int8))
        all_ids.extend(identities)
        best_epoch = int(result["best_epoch"])
        best_epochs.append(best_epoch)
        fold_geometry.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "checkpoint_sha256": result["best_checkpoint_sha256"],
                "training_windows_for_axis": len(train_rows),
                "oof_windows": len(validation_rows),
                "head_axis_cosine": cosine,
                "head_axis_angle_degrees": angle,
                "axis": axis.cpu().numpy().tolist(),
                "axis_stats": axis_stats,
            }
        )

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    folds = np.concatenate(all_folds)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(expected_ids):
        raise ValueError("Main M2 OOF exact identity coverage mismatch")
    if len(scores) != int(config["expected_source_windows"]):
        raise ValueError("Main M2 OOF window count mismatch")
    expected_labels = {identity: row.binary_label for identity, row in zip(expected_ids, expected_rows)}
    if any(expected_labels[identity] != int(label) for identity, label in zip(all_ids, labels)):
        raise ValueError("Main M2 OOF identity-label mismatch")
    arrays = {
        "labels": labels,
        "fold_id": folds,
        **_identity_arrays(all_ids),
        "head_logit_difference": scores,
    }
    archive_path = output / "oof_scores.npz"
    _write_npz(archive_path, arrays)
    curve, selected_threshold = threshold_curve(labels, scores, fixed_threshold=0.0)
    metrics = compute_binary_metrics(
        labels, scores, threshold=float(selected_threshold["threshold"])
    )
    summary = {
        "formal": not diagnostic,
        "target_data_accessed": False,
        "lambda_axis": float(lambda_axis),
        "windows": len(scores),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "optimized_threshold": float(selected_threshold["threshold"]),
        "optimized_metrics": metrics,
        "fold_angles": fold_geometry,
        "mean_head_axis_angle_degrees": float(
            np.mean([item["head_axis_angle_degrees"] for item in fold_geometry])
        ),
        "std_head_axis_angle_degrees": float(
            np.std([item["head_axis_angle_degrees"] for item in fold_geometry], ddof=0)
        ),
        "mean_head_axis_cosine": float(
            np.mean([item["head_axis_cosine"] for item in fold_geometry])
        ),
        "best_epochs": best_epochs,
        "archive_sha256": sha256_file(archive_path),
        "runtime_seconds": time.perf_counter() - started,
        "git": git,
        "environment": environment_snapshot(device),
    }
    write_json(output / "summary.json", summary)
    write_csv(
        output / "threshold_curve.csv",
        [{"lambda_axis": float(lambda_axis), **row} for row in curve],
    )
    write_json(output / "run_manifest.json", summary)
    return summary


def choose_lambda(candidate_rows: list[dict], selection: dict) -> tuple[dict | None, list[dict]]:
    tolerance = float(selection["numeric_tolerance"])
    minimum_auroc = float(selection["ce_auroc"]) - float(selection["max_auroc_drop"])
    minimum_bacc = float(selection["ce_optimized_bacc"]) - float(
        selection["max_bacc_drop"]
    )
    audited = []
    for row in candidate_rows:
        item = dict(row)
        item["auroc_margin_vs_floor"] = float(row["auroc"]) - minimum_auroc
        item["bacc_margin_vs_floor"] = float(row["balanced_accuracy"]) - minimum_bacc
        item["eligible"] = bool(
            float(row["auroc"]) + tolerance >= minimum_auroc
            and float(row["balanced_accuracy"]) + tolerance >= minimum_bacc
        )
        audited.append(item)
    eligible = [row for row in audited if row["eligible"]]
    if not eligible:
        return None, audited
    best_angle = min(float(row["mean_head_axis_angle_degrees"]) for row in eligible)
    tied = [
        row
        for row in eligible
        if np.isclose(
            float(row["mean_head_axis_angle_degrees"]),
            best_angle,
            atol=tolerance,
            rtol=0,
        )
    ]
    return min(tied, key=lambda row: float(row["lambda_axis"])), audited


def select_m2_lambda(config: dict, *, output_override: Path | None = None) -> dict:
    validate_m2_config(config)
    root = output_override or Path(config["output_dir"]) / "oof"
    diagnostic = output_override is not None
    git = git_identity()
    if not diagnostic and git.get("dirty") is not False:
        raise ValueError("formal Main M2 selection requires a clean Git worktree")
    selection_path = root / "selection_artifact.json"
    if selection_path.exists():
        raise FileExistsError("Main M2 selection artifact already exists")
    candidates = []
    for value in config["lambdas"]:
        directory = root / lambda_slug(float(value))
        summary_path = directory / "summary.json"
        summary = load_json(summary_path)
        archive_path = directory / "oof_scores.npz"
        if (
            (not diagnostic and not summary.get("formal"))
            or summary.get("target_data_accessed") is not False
            or sha256_file(archive_path) != summary.get("archive_sha256")
        ):
            raise ValueError(f"invalid Main M2 OOF candidate lambda={value}")
        candidates.append(
            {
                "lambda_axis": float(value),
                "auroc": float(summary["auroc"]),
                "auprc": float(summary["auprc"]),
                "balanced_accuracy": float(
                    summary["optimized_metrics"]["balanced_accuracy"]
                ),
                "macro_f1": float(summary["optimized_metrics"]["macro_f1"]),
                "mcc": float(summary["optimized_metrics"]["mcc"]),
                "optimized_threshold": float(summary["optimized_threshold"]),
                "mean_head_axis_cosine": float(summary["mean_head_axis_cosine"]),
                "mean_head_axis_angle_degrees": float(
                    summary["mean_head_axis_angle_degrees"]
                ),
                "std_head_axis_angle_degrees": float(
                    summary["std_head_axis_angle_degrees"]
                ),
                "best_epochs": summary["best_epochs"],
                "archive_sha256": summary["archive_sha256"],
            }
        )
    selected, audited = choose_lambda(candidates, config["selection"])
    write_csv(root / "lambda_metrics.csv", audited)
    artifact = {
        "formal": not diagnostic,
        "frozen": True,
        "target_data_accessed": False,
        "selection_rule": (
            "pass_R2_CE_AUROC_and_BACC_floors_then_min_mean_fold_angle_"
            "then_smaller_lambda"
        ),
        "ce_reference": {
            "auroc": config["selection"]["ce_auroc"],
            "auprc": config["selection"]["ce_auprc"],
            "optimized_bacc": config["selection"]["ce_optimized_bacc"],
            "archive_sha256": config["selection"]["r2_oof_archive_sha256"],
        },
        "candidates": audited,
        "status": "SELECTED" if selected is not None else "NO_ELIGIBLE_LAMBDA",
        "selected_lambda": None if selected is None else selected["lambda_axis"],
        "selected_threshold": (
            None if selected is None else selected["optimized_threshold"]
        ),
        "protocol_sha256": config["protocol_sha256"],
        "git": git,
    }
    write_json(selection_path, artifact)
    if selected is not None:
        final_epoch = median_best_epoch([int(x) for x in selected["best_epochs"]])
        epoch_rule = {
            "formal": not diagnostic,
            "frozen": True,
            "target_data_accessed": False,
            "selected_lambda": selected["lambda_axis"],
            "best_epochs": selected["best_epochs"],
            "final_epoch": final_epoch,
            "rule": config["full_source"]["epoch_rule"],
            "selection_artifact_sha256": sha256_file(selection_path),
        }
        write_json(root / "final_epoch_rule.json", epoch_rule)
        artifact["final_epoch"] = final_epoch
    return artifact
