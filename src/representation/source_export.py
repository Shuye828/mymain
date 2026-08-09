"""Export source-train embeddings, prototypes, and AF disease direction."""

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

from src.adaptation.disease_direction import SourcePrototypeAccumulator
from src.data.ecg_dataset import ECGWindowDataset, WindowRow, load_window_rows
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_feature_archive(
    path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    datasets: list[str],
    subject_ids: list[str],
    record_ids: list[str],
    window_starts: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            labels=labels,
            dataset=np.asarray(datasets, dtype=np.str_),
            subject_id=np.asarray(subject_ids, dtype=np.str_),
            record_id=np.asarray(record_ids, dtype=np.str_),
            window_start=window_starts,
        )
    temporary.replace(path)


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_source_rows(source_config: dict) -> list[WindowRow]:
    training = source_config["training"]
    return load_window_rows(
        [Path(source_config["index_path"])],
        source_split="train",
        max_windows_per_subject_per_class=int(
            training["max_windows_per_subject_per_class"]
        ),
        seed=int(training["seed"]),
    )


def _validate_checkpoint_provenance(
    payload: dict,
    *,
    dataset: str,
    index_path: Path,
) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("checkpoint has no provenance dictionary")
    checkpoint_config = provenance.get("config", {})
    if checkpoint_config.get("dataset") != dataset:
        raise ValueError("checkpoint dataset does not match export dataset")
    expected_hash = provenance.get("index_sha256")
    current_hash = sha256_file(index_path)
    if expected_hash != current_hash:
        raise ValueError("checkpoint index hash does not match current index")
    return current_hash


def export_source_direction(
    config: dict,
    *,
    device_request: str = "auto",
    output_override: Path | None = None,
    max_batches: int | None = None,
) -> dict:
    """Export deterministic source-train backbone embeddings and direction.

    This API accepts a source-direction configuration only. It always selects
    the visible source ``train`` split and has no target-domain argument.
    """

    config = deepcopy(config)
    if config.get("role") != "source_direction":
        raise ValueError("feature export requires role='source_direction'")
    if config.get("representation", {}).get("kind") != "backbone_l2":
        raise ValueError("formal source direction requires backbone_l2 features")
    source_config_path = Path(config["source_config"])
    source_config = _load_json(source_config_path)
    if source_config.get("role") != "source":
        raise ValueError("referenced training configuration is not source-only")
    if source_config.get("dataset") != config.get("dataset"):
        raise ValueError("source and direction configuration datasets differ")

    seed = int(source_config["training"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_request)
    rows = _load_source_rows(source_config)
    if not rows:
        raise ValueError("source training split contains no windows")
    dataset = ECGWindowDataset(
        rows,
        data_root=Path(source_config["data_root"]),
    )
    export_config = config["export"]
    loader = DataLoader(
        dataset,
        batch_size=int(export_config["batch_size"]),
        shuffle=False,
        num_workers=int(export_config["num_workers"]),
        pin_memory=False,
    )

    model = SourceMedTSTTT(**source_config["model"]).to(device)
    checkpoint_path = Path(config["checkpoint"])
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    index_path = Path(source_config["index_path"])
    index_hash = _validate_checkpoint_provenance(
        checkpoint,
        dataset=str(config["dataset"]),
        index_path=index_path,
    )

    feature_dim = int(source_config["model"]["dim"])
    expected_dim = int(config["representation"]["expected_dim"])
    if feature_dim != expected_dim:
        raise ValueError(
            f"configured feature dimension {expected_dim} does not match "
            f"model dimension {feature_dim}"
        )
    accumulator = SourcePrototypeAccumulator(feature_dim)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    datasets: list[str] = []
    subject_ids: list[str] = []
    record_ids: list[str] = []
    window_starts: list[int] = []
    progress_every = int(export_config.get("progress_every_batches", 250))
    if progress_every <= 0:
        raise ValueError("progress interval must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")

    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            labels = batch["y"]
            features = F.normalize(
                model.forward_features(x), dim=-1, eps=1e-12
            )
            if not torch.isfinite(features).all():
                raise FloatingPointError("feature export produced NaN or Inf")
            accumulator.update(features, labels)
            feature_chunks.append(features.cpu().numpy().astype(np.float32))
            label_chunks.append(labels.numpy().astype(np.int8))
            metadata = batch["metadata"]
            datasets.extend(str(value) for value in metadata["dataset"])
            subject_ids.extend(str(value) for value in metadata["subject_id"])
            record_ids.extend(str(value) for value in metadata["record_id"])
            starts = metadata["window_start"]
            window_starts.extend(int(value) for value in starts.tolist())
            if batch_index % progress_every == 0:
                print(
                    f"phase=source_feature_export batch={batch_index} "
                    f"samples={len(window_starts)} "
                    f"seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
            if max_batches is not None and batch_index >= max_batches:
                break

    features_array = np.concatenate(feature_chunks, axis=0)
    labels_array = np.concatenate(label_chunks, axis=0)
    starts_array = np.asarray(window_starts, dtype=np.int64)
    if not (
        len(features_array)
        == len(labels_array)
        == len(datasets)
        == len(subject_ids)
        == len(record_ids)
        == len(starts_array)
    ):
        raise RuntimeError("feature export arrays and metadata are misaligned")
    direction = accumulator.finalize()
    norm_error = float(
        np.max(np.abs(np.linalg.norm(features_array, axis=1) - 1.0))
    )
    projections = features_array @ direction.direction.numpy()
    nonaf_projection_mean = float(projections[labels_array == 0].mean())
    af_projection_mean = float(projections[labels_array == 1].mean())
    if not af_projection_mean > nonaf_projection_mean:
        raise RuntimeError("estimated disease direction has reversed source means")

    output_dir = output_override or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "source_train_features.npz"
    direction_path = output_dir / "disease_direction.json"
    manifest_path = output_dir / "run_manifest.json"
    result_path = output_dir / "result.json"
    _write_feature_archive(
        archive_path,
        features=features_array,
        labels=labels_array,
        datasets=datasets,
        subject_ids=subject_ids,
        record_ids=record_ids,
        window_starts=starts_array,
    )
    direction_payload = {
        "dataset": config["dataset"],
        "representation": config["representation"],
        "nonaf_prototype": direction.nonaf_prototype.tolist(),
        "af_prototype": direction.af_prototype.tolist(),
        "direction": direction.direction.tolist(),
        "nonaf_count": direction.nonaf_count,
        "af_count": direction.af_count,
        "direction_norm": float(torch.linalg.vector_norm(direction.direction)),
        "nonaf_projection_mean": nonaf_projection_mean,
        "af_projection_mean": af_projection_mean,
        "projection_mean_gap": af_projection_mean - nonaf_projection_mean,
    }
    _write_json(direction_path, direction_payload)
    runtime_seconds = time.perf_counter() - started
    manifest = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "config": config,
        "source_config": source_config,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "index_sha256": index_hash,
        "source_split": "train",
        "source_window_count": len(features_array),
        "diagnostic_max_batches": max_batches,
    }
    _write_json(manifest_path, manifest)
    result = {
        "dataset": config["dataset"],
        "source_split": "train",
        "window_count": len(features_array),
        "feature_dim": feature_dim,
        "feature_norm_max_abs_error": norm_error,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "runtime_seconds": runtime_seconds,
        "archive_path": str(archive_path),
        "direction_path": str(direction_path),
        "direction": direction_payload,
    }
    _write_json(result_path, result)
    return result
