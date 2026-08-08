"""Orchestrate CE-only source training with frozen protocol provenance."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    ECGWindowDataset,
    WindowRow,
    build_subject_class_balanced_sampler,
    load_window_rows,
)
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.engine import evaluate, train_one_epoch
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _tiny_rows(rows: list[WindowRow], per_class: int) -> list[WindowRow]:
    selected: dict[int, list[WindowRow]] = {0: [], 1: []}
    used_subjects: set[tuple[int, str]] = set()
    for row in rows:
        label = row.binary_label
        if len(selected[label]) >= per_class:
            continue
        key = (label, row.subject_id)
        if key not in used_subjects or len(selected[label]) >= 4:
            selected[label].append(row)
            used_subjects.add(key)
        if all(len(items) >= per_class for items in selected.values()):
            break
    result = selected[0] + selected[1]
    if len(selected[0]) < per_class or len(selected[1]) < per_class:
        raise ValueError("could not construct balanced tiny subset")
    return result


def _balanced_diagnostic_rows(
    rows: list[WindowRow], *, per_class: int, seed: int
) -> list[WindowRow]:
    """Select a deterministic source-only diagnostic subset by class."""

    if per_class <= 0:
        raise ValueError("diagnostic class cap must be positive")
    selected: list[WindowRow] = []
    for label in (0, 1):
        candidates = [row for row in rows if row.binary_label == label]
        candidates.sort(
            key=lambda row: (
                _row_priority_for_diagnostic(row, seed),
                row.record_id,
                row.start_sample,
            )
        )
        if len(candidates) < per_class:
            raise ValueError(
                f"class {label} has {len(candidates)} rows, fewer than {per_class}"
            )
        selected.extend(candidates[:per_class])
    selected.sort(
        key=lambda row: (
            row.binary_label,
            row.subject_id,
            row.record_id,
            row.start_sample,
        )
    )
    return selected


def _row_priority_for_diagnostic(row: WindowRow, seed: int) -> int:
    payload = (
        f"diagnostic:{seed}:{row.dataset}:{row.subject_id}:"
        f"{row.record_id}:{row.start_sample}:{row.binary_label}"
    )
    return int.from_bytes(
        hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big"
    )


def _build_loaders(
    config: dict,
    *,
    tiny_overfit: bool,
    evaluation_windows_per_class: int | None = None,
) -> tuple:
    dataset = config["dataset"]
    index_path = Path(config["index_path"])
    data_root = Path(config["data_root"])
    training = config["training"]
    seed = int(training["seed"])
    train_rows = load_window_rows(
        [index_path],
        source_split="train",
        max_windows_per_subject_per_class=int(
            training["max_windows_per_subject_per_class"]
        ),
        seed=seed,
    )
    if tiny_overfit:
        train_rows = _tiny_rows(train_rows, int(training["tiny_per_class"]))
        validation_rows = train_rows
        test_rows: list[WindowRow] = []
    else:
        validation_rows = load_window_rows(
            [index_path], source_split="validation"
        )
        test_rows = load_window_rows([index_path], source_split="test")
        if evaluation_windows_per_class is not None:
            validation_rows = _balanced_diagnostic_rows(
                validation_rows,
                per_class=evaluation_windows_per_class,
                seed=seed,
            )
            test_rows = _balanced_diagnostic_rows(
                test_rows,
                per_class=evaluation_windows_per_class,
                seed=seed,
            )

    train_dataset = ECGWindowDataset(train_rows, data_root=data_root)
    validation_dataset = ECGWindowDataset(validation_rows, data_root=data_root)
    test_dataset = ECGWindowDataset(test_rows, data_root=data_root)
    train_sampler = build_subject_class_balanced_sampler(
        train_rows,
        seed=seed,
        num_samples=(
            len(train_rows)
            if not tiny_overfit
            else max(len(train_rows), int(training["batch_size"]))
        ),
    )
    loader_kwargs = {
        "num_workers": int(training["num_workers"]),
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        sampler=train_sampler,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=int(training["eval_batch_size"]),
            shuffle=False,
            **loader_kwargs,
        )
        if test_rows
        else None
    )
    counts = {
        "dataset": dataset,
        "train_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "test_windows": len(test_rows),
        "tiny_overfit": tiny_overfit,
        "evaluation_windows_per_class": evaluation_windows_per_class,
    }
    return train_loader, validation_loader, test_loader, counts


def train_source_experiment(
    config: dict,
    *,
    tiny_overfit: bool = False,
    device_request: str = "auto",
    epoch_override: int | None = None,
    output_override: Path | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    resume_path: Path | None = None,
    evaluation_windows_per_class: int | None = None,
) -> dict:
    """Train one source model; target data are never accepted by this API."""

    config = deepcopy(config)
    if config.get("role") != "source":
        raise ValueError("stage-3 trainer accepts source configurations only")
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(device_request)
    output_dir = output_override or Path(config["output_dir"])
    if tiny_overfit and output_override is None:
        output_dir = output_dir.parent / f"tiny_{config['dataset']}_ce"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = _build_loaders(
        config,
        tiny_overfit=tiny_overfit,
        evaluation_windows_per_class=evaluation_windows_per_class,
    )
    train_loader, validation_loader, test_loader, data_counts = loaders
    model = SourceMedTSTTT(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    max_epochs = int(epoch_override or training["epochs"])
    early_stopping = EarlyStopping(patience=int(training["patience"]))
    provenance = {
        "git": git_identity(),
        "environment": environment_snapshot(device),
        "index_path": config["index_path"],
        "index_sha256": sha256_file(Path(config["index_path"])),
        "data_counts": data_counts,
        "config": config,
    }

    history: list[dict] = []
    start_epoch = 1
    resumed_from: str | None = None
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        if resume_path.parent.resolve() != output_dir.resolve():
            raise ValueError(
                "resume checkpoint must be inside the selected output directory"
            )
        resume_payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(resume_payload["epoch"]) + 1
        history = list(resume_payload.get("history", []))
        early_stopping.best = float(
            resume_payload.get("best_validation_macro_f1", float("-inf"))
        )
        early_stopping.bad_epochs = int(
            resume_payload.get("early_stopping_bad_epochs", 0)
        )
        resumed_from = str(resume_path)
        provenance["resumed_from"] = resumed_from
        provenance["resume_epoch"] = int(resume_payload["epoch"])
    if start_epoch > max_epochs:
        raise ValueError(
            f"checkpoint is at epoch {start_epoch - 1}, but max epochs is {max_epochs}"
        )
    _write_json(output_dir / "run_manifest.json", provenance)

    start_time = time.perf_counter()
    if resume_payload is not None and "initial_validation" in resume_payload:
        initial_validation = resume_payload["initial_validation"]
    else:
        initial_validation = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_eval_batches,
        )
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_train_batches,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        improved, should_stop = early_stopping.update(
            float(validation_metrics["macro_f1"])
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_macro_f1": early_stopping.best,
            "early_stopping_bad_epochs": early_stopping.bad_epochs,
            "initial_validation": initial_validation,
            "history": history,
            "provenance": provenance,
        }
        save_checkpoint(last_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        _write_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_f1={validation_metrics['macro_f1']:.4f} "
            f"seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if not tiny_overfit and should_stop:
            break

    best_payload = load_checkpoint(best_path, model=model, map_location=device)
    best_validation = evaluate(
        model,
        validation_loader,
        criterion=criterion,
        device=device,
        max_batches=max_eval_batches,
    )
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            criterion=criterion,
            device=device,
            max_batches=max_eval_batches,
        )
    result = {
        "dataset": config["dataset"],
        "tiny_overfit": tiny_overfit,
        "resumed_from": resumed_from,
        "epochs_completed": len(history),
        "best_epoch": int(best_payload["epoch"]),
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "test": test_metrics,
        "runtime_seconds": time.perf_counter() - start_time,
        "output_dir": str(output_dir),
        "data_counts": data_counts,
    }
    if tiny_overfit:
        result["tiny_success"] = bool(
            best_validation["accuracy"] >= float(training["tiny_success_accuracy"])
            and best_validation["loss"] < initial_validation["loss"]
        )
    _write_json(output_dir / "result.json", result)
    return result
