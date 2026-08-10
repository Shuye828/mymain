"""Finalize Revision R2 AFDB out-of-fold scores and thresholds."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ecg_dataset import ECGWindowDataset, WindowRow, load_window_rows
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import threshold_curve
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.afdb_source_protocol import (
    centered_prototype_margin,
    config_protocol_hash,
    fold_subject_partitions,
    median_best_epoch,
    read_fold_assignments,
    write_json,
)
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


def row_identity(row: WindowRow) -> tuple[str, str, str, int]:
    return row.dataset, row.subject_id, row.record_id, row.start_sample


def validate_oof_coverage(
    expected: list[WindowRow], identities: list[tuple[str, str, str, int]]
) -> None:
    expected_ids = [row_identity(row) for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("current AFDB index contains duplicate window identities")
    if len(identities) != len(set(identities)):
        raise ValueError("OOF archive contains duplicate window identities")
    if set(identities) != set(expected_ids):
        missing = len(set(expected_ids) - set(identities))
        extra = len(set(identities) - set(expected_ids))
        raise ValueError(f"OOF coverage mismatch: missing={missing}, extra={extra}")


def _extract(
    model: SourceMedTSTTT,
    rows: list[WindowRow],
    *,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str, str, int]], np.ndarray]:
    loader = DataLoader(
        ECGWindowDataset(rows, data_root=data_root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    feature_chunks, label_chunks, head_chunks = [], [], []
    identities = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits, features = model(batch["x"].to(device), return_features=True)
            features = F.normalize(features, dim=-1, eps=1e-12)
            if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
                raise FloatingPointError("OOF extraction produced NaN/Inf")
            feature_chunks.append(features.cpu().numpy().astype(np.float32))
            label_chunks.append(batch["y"].numpy().astype(np.int8))
            head_chunks.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            metadata = batch["metadata"]
            for dataset, subject, record, start in zip(
                metadata["dataset"],
                metadata["subject_id"],
                metadata["record_id"],
                metadata["window_start"].tolist(),
            ):
                identities.append((str(dataset), str(subject), str(record), int(start)))
    return (
        np.concatenate(feature_chunks),
        np.concatenate(label_chunks),
        identities,
        np.concatenate(head_chunks).astype(np.float64),
    )


def finalize_afdb_oof(
    config: dict, *, device_request: str = "auto", output_override: Path | None = None
) -> dict:
    config_protocol_hash(config)
    if config.get("role") != "afdb_source_oof":
        raise ValueError("OOF finalization requires the frozen R2 config")
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_request)
    index_path = Path(config["index_path"])
    fold_path = Path(config["fold_manifest"])
    assignments = read_fold_assignments(fold_path)
    all_rows = load_window_rows([index_path])
    output = output_override or Path(config["output_dir"]) / "oof"
    output.mkdir(parents=True, exist_ok=True)
    if output_override is None and any(output.iterdir()):
        raise FileExistsError(f"formal OOF output is not empty: {output}")

    arrays = {key: [] for key in ("labels", "fold_id", "head", "prototype")}
    all_identities = []
    fold_summaries = []
    best_epochs = []
    started = time.perf_counter()
    for fold_id in range(5):
        training_subjects, validation_subjects = fold_subject_partitions(
            assignments, fold_id
        )
        fold_dir = Path(config["output_dir"]) / "folds" / f"fold_{fold_id}"
        checkpoint_path = fold_dir / "best.pt"
        result = json.loads((fold_dir / "result.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        provenance = checkpoint.get("provenance", {})
        checkpoint_config = provenance.get("config", {})
        if checkpoint_config.get("r2_fold_id") != fold_id:
            raise ValueError(f"fold {fold_id} checkpoint provenance mismatch")
        if provenance.get("index_sha256") != sha256_file(index_path):
            raise ValueError(f"fold {fold_id} index hash mismatch")
        if checkpoint_config.get("fold_manifest_sha256") != sha256_file(fold_path):
            raise ValueError(f"fold {fold_id} manifest hash mismatch")
        if set(checkpoint_config["subject_partitions"]["train"]) != training_subjects:
            raise ValueError(f"fold {fold_id} training subjects mismatch")
        if (
            set(checkpoint_config["subject_partitions"]["validation"])
            != validation_subjects
        ):
            raise ValueError(f"fold {fold_id} validation subjects mismatch")

        model = SourceMedTSTTT(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        train_rows = load_window_rows(
            [index_path],
            include_subjects=training_subjects,
            max_windows_per_subject_per_class=int(
                config["training"]["max_windows_per_subject_per_class"]
            ),
            seed=seed,
        )
        validation_rows = load_window_rows(
            [index_path], include_subjects=validation_subjects
        )
        train_features, train_labels, _, _ = _extract(
            model,
            train_rows,
            data_root=Path(config["data_root"]),
            batch_size=int(config["training"]["eval_batch_size"]),
            num_workers=int(config["training"]["num_workers"]),
            device=device,
        )
        direction, _, midpoint = centered_prototype_margin(train_features, train_labels)
        features, labels, identities, head = _extract(
            model,
            validation_rows,
            data_root=Path(config["data_root"]),
            batch_size=int(config["training"]["eval_batch_size"]),
            num_workers=int(config["training"]["num_workers"]),
            device=device,
        )
        prototype = features @ direction.astype(np.float32) - midpoint
        if any(identity[1] in training_subjects for identity in identities):
            raise ValueError(f"fold {fold_id} OOF prediction leaks a training subject")
        arrays["labels"].append(labels)
        arrays["fold_id"].append(np.full(len(labels), fold_id, dtype=np.int8))
        arrays["head"].append(head)
        arrays["prototype"].append(prototype.astype(np.float64))
        all_identities.extend(identities)
        best_epoch = int(result["best_epoch"])
        best_epochs.append(best_epoch)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "best_epoch": best_epoch,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "train_windows_for_prototype": len(train_rows),
                "oof_windows": len(validation_rows),
                "prototype_midpoint": midpoint,
                "direction": direction.tolist(),
            }
        )

    validate_oof_coverage(all_rows, all_identities)
    labels = np.concatenate(arrays["labels"])
    fold_ids = np.concatenate(arrays["fold_id"])
    scores = {
        "head_logit_difference": np.concatenate(arrays["head"]),
        "prototype_margin": np.concatenate(arrays["prototype"]),
    }
    archive_path = output / "afdb_oof_scores.npz"
    with archive_path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=labels,
            fold_id=fold_ids,
            dataset=np.asarray([identity[0] for identity in all_identities]),
            subject_id=np.asarray([identity[1] for identity in all_identities]),
            record_id=np.asarray([identity[2] for identity in all_identities]),
            window_start=np.asarray([identity[3] for identity in all_identities]),
            **scores,
        )
    archive_path.with_suffix(".npz.tmp").replace(archive_path)

    curve_rows, thresholds, metric_rows = [], {}, []
    for score_name, values in scores.items():
        curve, selected = threshold_curve(labels, values, fixed_threshold=0.0)
        thresholds[score_name] = {"fixed": 0.0, "optimized": selected}
        for row in curve:
            curve_rows.append({"score": score_name, **row})
        for baseline, threshold in (
            ("fixed", 0.0),
            ("optimized", selected["threshold"]),
        ):
            metric_rows.append(
                {
                    "score": score_name,
                    "baseline": baseline,
                    **compute_binary_metrics(labels, values, threshold=threshold),
                }
            )
    with (output / "threshold_curves.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "fold_score_summaries.json", {"folds": fold_summaries})
    write_json(output / "oof_metrics.json", {"metrics": metric_rows})
    final_epoch = median_best_epoch(best_epochs)
    final_rule = {
        "frozen": True,
        "best_epochs": best_epochs,
        "final_epoch": final_epoch,
        "rule": "integer_median_of_five_oof_best_epochs",
        "target_data_accessed": False,
        "oof_archive_sha256": sha256_file(archive_path),
    }
    write_json(Path(config["output_dir"]) / "final_epoch_rule.json", final_rule)
    manifest = {
        "formal": output_override is None,
        "target_data_accessed": False,
        "source_labels_used": True,
        "oof_windows": len(labels),
        "fold_counts": dict(Counter(int(value) for value in fold_ids)),
        "index_sha256": sha256_file(index_path),
        "fold_manifest_sha256": sha256_file(fold_path),
        "archive_sha256": sha256_file(archive_path),
        "final_epoch": final_epoch,
        "runtime_seconds": time.perf_counter() - started,
        "git": git_identity(),
        "environment": environment_snapshot(device),
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest
