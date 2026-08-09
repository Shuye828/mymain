"""Label-free target scoring and frozen GMM adaptation workflow."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.adaptation.target_gmm import apply_gmm_boundary, fit_target_gmm
from src.data.ecg_dataset import ECGWindowDataset, load_unlabeled_target_rows
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
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


def _write_score_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if any(key in arrays for key in ("label", "labels", "binary_label")):
        raise ValueError("unlabeled target archive cannot contain labels")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("target score arrays are not aligned")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_hidden_target_rows(
    index_path: Path,
    *,
    max_windows_per_split: int | None = None,
) -> tuple[list, dict[str, Any]]:
    adaptation = load_unlabeled_target_rows(
        [index_path], target_split="adaptation"
    )
    evaluation = load_unlabeled_target_rows(
        [index_path], target_split="evaluation"
    )
    adaptation_subjects = {row.subject_id for row in adaptation}
    evaluation_subjects = {row.subject_id for row in evaluation}
    overlap = adaptation_subjects & evaluation_subjects
    if overlap:
        raise ValueError(
            f"target inductive subject leakage: {sorted(overlap)[:5]}"
        )
    full_adaptation_count = len(adaptation)
    full_evaluation_count = len(evaluation)
    if max_windows_per_split is not None:
        if max_windows_per_split < 4:
            raise ValueError("diagnostic split cap must be at least 4")
        adaptation = adaptation[:max_windows_per_split]
        evaluation = evaluation[:max_windows_per_split]
    rows = adaptation + evaluation
    if not rows or any(row.binary_label != -1 for row in rows):
        raise ValueError("target workflow did not receive hidden labels")
    return rows, {
        "adaptation_windows": len(adaptation),
        "evaluation_windows": len(evaluation),
        "transductive_windows": len(rows),
        "full_adaptation_windows": full_adaptation_count,
        "full_evaluation_windows": full_evaluation_count,
        "adaptation_subjects": len(adaptation_subjects),
        "evaluation_subjects": len(evaluation_subjects),
        "subject_overlap": 0,
    }


def _validate_source_artifacts(
    config: dict,
    *,
    source_config: dict,
    checkpoint_path: Path,
    direction_path: Path,
) -> tuple[dict, dict]:
    direction = _load_json(direction_path)
    direction_manifest_path = direction_path.parent / "run_manifest.json"
    direction_manifest = _load_json(direction_manifest_path)
    if direction.get("dataset") != config.get("source_dataset"):
        raise ValueError("disease direction source dataset mismatch")
    if source_config.get("dataset") != config.get("source_dataset"):
        raise ValueError("source model configuration dataset mismatch")
    if direction.get("representation", {}).get("kind") != "backbone_l2":
        raise ValueError("target GMM requires a frozen backbone_l2 direction")
    if direction_manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("source checkpoint does not match direction provenance")
    if direction_manifest.get("diagnostic_max_batches") is not None:
        raise ValueError("target GMM cannot use a diagnostic source direction")
    vector = np.asarray(direction.get("direction"), dtype=np.float64)
    expected_dim = int(source_config["model"]["dim"])
    if vector.shape != (expected_dim,) or not np.isfinite(vector).all():
        raise ValueError("disease direction has an invalid shape or values")
    if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5):
        raise ValueError("disease direction is not unit normalized")
    return direction, direction_manifest


def fit_target_gmm_experiment(
    config: dict,
    *,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches: int | None = None,
) -> dict:
    """Score hidden-label target inputs and freeze both target GMM protocols."""

    config = deepcopy(config)
    if config.get("role") != "target_gmm":
        raise ValueError("target workflow requires role='target_gmm'")
    if max_batches is not None and output_override is None:
        raise ValueError(
            "diagnostic max_batches requires an explicit output override"
        )
    protocols = set(config.get("protocols", []))
    if protocols != {"inductive_holdout", "transductive"}:
        raise ValueError("both target protocols must be explicitly enabled")
    source_config_path = Path(config["source_config"])
    source_config = _load_json(source_config_path)
    if source_config.get("role") != "source":
        raise ValueError("referenced model configuration is not source-only")
    checkpoint_path = Path(config["checkpoint"])
    direction_path = Path(config["direction"])
    direction, direction_manifest = _validate_source_artifacts(
        config,
        source_config=source_config,
        checkpoint_path=checkpoint_path,
        direction_path=direction_path,
    )
    target_index_path = Path(config["target_index"])
    batch_size = int(config["extraction"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("extraction batch size must be positive")
    diagnostic_split_cap = None
    if max_batches is not None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        diagnostic_split_cap = max(4, max_batches * batch_size // 2)
    rows, target_counts = _load_hidden_target_rows(
        target_index_path,
        max_windows_per_split=diagnostic_split_cap,
    )
    observed_datasets = {row.dataset for row in rows}
    if observed_datasets != {config["target_dataset"]}:
        raise ValueError(
            "target index dataset mismatch: "
            f"expected {config['target_dataset']}, got {sorted(observed_datasets)}"
        )

    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(device_request)
    model = SourceMedTSTTT(**source_config["model"]).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    direction_tensor = torch.tensor(
        direction["direction"], dtype=torch.float32, device=device
    )
    dataset = ECGWindowDataset(
        rows,
        data_root=Path(config["target_data_root"]),
        expose_label=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["extraction"]["num_workers"]),
        pin_memory=False,
    )
    progress_every = int(
        config["extraction"].get("progress_every_batches", 250)
    )
    if progress_every <= 0:
        raise ValueError("progress interval must be positive")
    datasets: list[str] = []
    subject_ids: list[str] = []
    record_ids: list[str] = []
    window_starts: list[int] = []
    target_splits: list[str] = []
    classifier_chunks: list[np.ndarray] = []
    direction_chunks: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            if not bool((batch["y"] == -1).all()):
                raise ValueError("target loader exposed a label")
            inputs = batch["x"].to(device)
            logits, features = model(inputs, return_features=True)
            features = F.normalize(features, dim=-1, eps=1e-12)
            classifier_probability = torch.softmax(logits, dim=1)[:, 1]
            direction_score = features @ direction_tensor
            if (
                not torch.isfinite(classifier_probability).all()
                or not torch.isfinite(direction_score).all()
            ):
                raise FloatingPointError("target scoring produced NaN or Inf")
            classifier_chunks.append(
                classifier_probability.cpu().numpy().astype(np.float32)
            )
            direction_chunks.append(
                direction_score.cpu().numpy().astype(np.float32)
            )
            metadata = batch["metadata"]
            datasets.extend(str(value) for value in metadata["dataset"])
            subject_ids.extend(str(value) for value in metadata["subject_id"])
            record_ids.extend(str(value) for value in metadata["record_id"])
            window_starts.extend(
                int(value) for value in metadata["window_start"].tolist()
            )
            target_splits.extend(
                str(value) for value in metadata["target_split"]
            )
            if batch_index % progress_every == 0:
                print(
                    f"phase=target_unlabeled_scoring batch={batch_index} "
                    f"samples={len(window_starts)} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    classifier_probability = np.concatenate(classifier_chunks)
    direction_scores = np.concatenate(direction_chunks)
    split_array = np.asarray(target_splits, dtype=np.str_)
    sample_count = len(direction_scores)
    if not (
        sample_count
        == len(classifier_probability)
        == len(datasets)
        == len(subject_ids)
        == len(record_ids)
        == len(window_starts)
        == len(split_array)
    ):
        raise RuntimeError("target scores and metadata are misaligned")
    if not set(split_array.tolist()).issubset({"adaptation", "evaluation"}):
        raise ValueError("target score archive contains an unknown split")
    adaptation_mask = split_array == "adaptation"
    evaluation_mask = split_array == "evaluation"
    if not adaptation_mask.any() or not evaluation_mask.any():
        raise ValueError("both inductive target splits must be present")

    gmm_config = config["gmm"]
    reliability = config["reliability"]
    print(
        f"phase=fit_inductive_gmm samples={int(adaptation_mask.sum())}",
        flush=True,
    )
    inductive = fit_target_gmm(
        direction_scores[adaptation_mask],
        random_state=seed,
        n_init=int(gmm_config["n_init"]),
        reg_covar=float(gmm_config["reg_covar"]),
        stability_runs=int(gmm_config["stability_runs"]),
        reliability=reliability,
    )
    print(f"phase=fit_transductive_gmm samples={sample_count}", flush=True)
    transductive = fit_target_gmm(
        direction_scores,
        random_state=seed,
        n_init=int(gmm_config["n_init"]),
        reg_covar=float(gmm_config["reg_covar"]),
        stability_runs=int(gmm_config["stability_runs"]),
        reliability=reliability,
    )
    inductive_probability = apply_gmm_boundary(direction_scores, inductive)
    transductive_probability = apply_gmm_boundary(direction_scores, transductive)
    source_threshold = float(direction["source_fixed_threshold"])
    inductive["source_fixed_threshold"] = source_threshold
    inductive["target_minus_source_threshold"] = (
        inductive["density_intersection_threshold"] - source_threshold
    )
    transductive["source_fixed_threshold"] = source_threshold
    transductive["target_minus_source_threshold"] = (
        transductive["density_intersection_threshold"] - source_threshold
    )

    output_dir = output_override or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "target_scores.npz"
    gmm_path = output_dir / "gmm_artifact.json"
    manifest_path = output_dir / "run_manifest.json"
    result_path = output_dir / "fit_result.json"
    arrays = {
        "dataset": np.asarray(datasets, dtype=np.str_),
        "subject_id": np.asarray(subject_ids, dtype=np.str_),
        "record_id": np.asarray(record_ids, dtype=np.str_),
        "window_start": np.asarray(window_starts, dtype=np.int64),
        "target_split": split_array,
        "source_classifier_probability": classifier_probability,
        "direction_score": direction_scores,
        "inductive_gmm_af_probability": inductive_probability.astype(
            np.float32
        ),
        "inductive_gmm_prediction": (inductive_probability >= 0.5).astype(np.int8),
        "transductive_gmm_af_probability": transductive_probability.astype(
            np.float32
        ),
        "transductive_gmm_prediction": (transductive_probability >= 0.5).astype(np.int8),
    }
    _write_score_archive(score_path, arrays)
    score_hash = sha256_file(score_path)
    gmm_artifact = {
        "frozen": True,
        "labels_accessed": False,
        "source_dataset": config["source_dataset"],
        "target_dataset": config["target_dataset"],
        "direction_sha256": sha256_file(direction_path),
        "target_index_sha256": sha256_file(target_index_path),
        "target_score_sha256": score_hash,
        "protocols": {
            "inductive_holdout": {
                "fit_split": "adaptation",
                "fit_count": int(adaptation_mask.sum()),
                "evaluation_split": "evaluation",
                "evaluation_count": int(evaluation_mask.sum()),
                "gmm": inductive,
            },
            "transductive": {
                "fit_split": "transductive_all",
                "fit_count": sample_count,
                "evaluation_split": "transductive_all",
                "evaluation_count": sample_count,
                "gmm": transductive,
            },
        },
    }
    _write_json(gmm_path, gmm_artifact)
    manifest = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "config": config,
        "source_config": source_config,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_direction_manifest": direction_manifest,
        "source_direction_sha256": sha256_file(direction_path),
        "target_index_sha256": sha256_file(target_index_path),
        "target_counts": target_counts,
        "labels_accessed": False,
        "diagnostic_max_batches": max_batches,
        "target_score_sha256": score_hash,
        "gmm_artifact_sha256": sha256_file(gmm_path),
    }
    _write_json(manifest_path, manifest)
    result = {
        "source_dataset": config["source_dataset"],
        "target_dataset": config["target_dataset"],
        "labels_accessed": False,
        "sample_count": sample_count,
        "target_counts": target_counts,
        "runtime_seconds": time.perf_counter() - started,
        "score_path": str(score_path),
        "gmm_artifact_path": str(gmm_path),
        "protocols": {
            name: {
                "fit_count": payload["fit_count"],
                "evaluation_count": payload["evaluation_count"],
                "reliable": payload["gmm"]["reliable"],
                "reliability_failures": payload["gmm"]["reliability_failures"],
                "delta_bic": payload["gmm"]["delta_bic"],
                "pooled_separation": payload["gmm"]["pooled_separation"],
                "normalized_posterior_entropy": payload["gmm"][
                    "normalized_posterior_entropy"
                ],
                "density_intersection_threshold": payload["gmm"][
                    "density_intersection_threshold"
                ],
            }
            for name, payload in gmm_artifact["protocols"].items()
        },
    }
    _write_json(result_path, result)
    return result
