"""Source-only epoch-level disease-axis alignment for Main M2."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    ECGWindowDataset,
    WindowRow,
    build_subject_class_balanced_sampler,
    load_window_rows,
)
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.afdb_source_protocol import (
    fold_subject_partitions,
    read_fold_assignments,
    validate_fold_assignments,
)
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.engine import evaluate
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)
from src.training.train_source import _balanced_diagnostic_rows


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


def lambda_slug(value: float) -> str:
    return f"lambda_{float(value):.2f}".replace(".", "p")


def validate_m2_config(config: dict) -> None:
    if config.get("role") != "source_axis_alignment":
        raise ValueError("Main M2 requires role='source_axis_alignment'")
    if config.get("dataset") != "afdb" or "targets" in config:
        raise ValueError("Main M2 accepts AFDB source data only")
    for key, hash_key in (
        ("protocol", "protocol_sha256"),
        ("governing_plan", "governing_plan_sha256"),
        ("r2_config", "r2_config_sha256"),
        ("index_path", "index_sha256"),
        ("fold_manifest", "fold_manifest_sha256"),
    ):
        if sha256_file(Path(config[key])) != config[hash_key]:
            raise ValueError(f"Main M2 frozen input changed: {key}")
    selection = config["selection"]
    for key in ("r2_oof_archive", "r2_oof_metrics"):
        if sha256_file(Path(selection[key])) != selection[f"{key}_sha256"]:
            raise ValueError(f"Main M2 frozen R2 reference changed: {key}")
    reference_rows = load_json(Path(selection["r2_oof_metrics"]))["metrics"]
    reference = next(
        (
            row
            for row in reference_rows
            if row.get("score") == "head_logit_difference"
            and row.get("baseline") == "optimized"
        ),
        None,
    )
    if reference is None or any(
        not np.isclose(float(reference[field]), float(selection[config_key]), atol=1e-12, rtol=0)
        for field, config_key in (
            ("auroc", "ce_auroc"),
            ("auprc", "ce_auprc"),
            ("balanced_accuracy", "ce_optimized_bacc"),
        )
    ):
        raise ValueError("Main M2 CE reference values do not match frozen R2 metrics")
    if list(map(float, config["lambdas"])) != [0.01, 0.05, 0.10, 0.20]:
        raise ValueError("Main M2 lambda grid changed")
    if int(config["folds"]) != 5 or int(config["seed"]) != 42:
        raise ValueError("Main M2 fold or seed identity changed")
    training = config["training"]
    if (
        training.get("axis_refresh")
        != "epoch_start_full_capped_training_cohort"
        or training.get("normalize_features_for_axis") is not True
        or training.get("checkpoint_metric") != "validation_macro_f1"
    ):
        raise ValueError("Main M2 training protocol changed")


def unit_vector(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError(f"{name} must be a finite nonzero vector")
    return vector / norm


def binary_head_direction(model: SourceMedTSTTT) -> torch.Tensor:
    weight = model.backbone.classification_head.weight
    return F.normalize(weight[1] - weight[0], dim=0, eps=1e-12)


def axis_alignment_loss(
    model: SourceMedTSTTT, frozen_axis: torch.Tensor
) -> torch.Tensor:
    if frozen_axis.ndim != 1 or frozen_axis.requires_grad:
        raise ValueError("axis loss requires a detached one-dimensional axis")
    if not torch.isfinite(frozen_axis).all():
        raise ValueError("axis loss received a nonfinite axis")
    direction = binary_head_direction(model)
    axis = F.normalize(frozen_axis, dim=0, eps=1e-12)
    loss = 1.0 - torch.sum(direction * axis)
    if not torch.isfinite(loss):
        raise FloatingPointError("axis alignment loss is nonfinite")
    return loss


def head_axis_geometry(
    model: SourceMedTSTTT, axis: torch.Tensor
) -> tuple[float, float]:
    with torch.no_grad():
        cosine = float(
            torch.clamp(
                torch.sum(binary_head_direction(model) * axis), -1.0, 1.0
            ).item()
        )
    return cosine, float(math.degrees(math.acos(cosine)))


@torch.no_grad()
def extract_source_axis(
    model: SourceMedTSTTT,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: int | None = None,
    progress_every_batches: int | None = None,
    phase: str = "source_axis",
) -> tuple[torch.Tensor, dict]:
    """Estimate a detached axis from visible AFDB source labels."""

    was_training = model.training
    model.eval()
    sums: dict[int, np.ndarray | None] = {0: None, 1: None}
    counts = {0: 0, 1: 0}
    batches = 0
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        labels = batch["y"]
        if not torch.isin(labels, torch.tensor([0, 1])).all():
            raise ValueError("source axis requires visible binary AFDB labels")
        features = F.normalize(
            model.forward_features(batch["x"].to(device)), dim=-1, eps=1e-12
        )
        if not torch.isfinite(features).all():
            raise FloatingPointError("source axis extraction produced NaN/Inf")
        features_np = features.cpu().numpy().astype(np.float64)
        labels_np = labels.numpy()
        for label in (0, 1):
            selected = features_np[labels_np == label]
            if len(selected):
                value = selected.sum(axis=0)
                sums[label] = value if sums[label] is None else sums[label] + value
                counts[label] += len(selected)
        batches += 1
        if progress_every_batches and batches % progress_every_batches == 0:
            print(
                f"phase={phase} batch={batches} samples={sum(counts.values())} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    if was_training:
        model.train()
    if min(counts.values()) <= 0 or sums[0] is None or sums[1] is None:
        raise ValueError("source axis extraction requires both source classes")
    prototype_0 = sums[0] / counts[0]
    prototype_1 = sums[1] / counts[1]
    direction = unit_vector(prototype_1 - prototype_0, name="source axis")
    axis = torch.as_tensor(direction, dtype=torch.float32, device=device).detach()
    cosine, angle = head_axis_geometry(model, axis)
    return axis, {
        "class_counts": {"0": counts[0], "1": counts[1]},
        "windows": sum(counts.values()),
        "batches": batches,
        "direction_norm": float(np.linalg.norm(direction)),
        "head_axis_cosine": cosine,
        "head_axis_angle_degrees": angle,
        "runtime_seconds": time.perf_counter() - started,
    }


def train_axis_epoch(
    model: SourceMedTSTTT,
    loader: Iterable,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    frozen_axis: torch.Tensor,
    lambda_axis: float,
    device: torch.device,
    max_batches: int | None = None,
    progress_every_batches: int | None = None,
    phase: str = "axis_train",
) -> dict:
    if lambda_axis <= 0:
        raise ValueError("Main M2 lambda must be positive")
    model.train()
    totals = {"ce": 0.0, "axis": 0.0, "total": 0.0}
    correct = samples = batches = 0
    grad_norm_sum = 0.0
    started = time.perf_counter()
    start_cosine, start_angle = head_axis_geometry(model, frozen_axis)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        ce_loss = criterion(logits, labels)
        axis_loss = axis_alignment_loss(model, frozen_axis)
        total_loss = ce_loss + float(lambda_axis) * axis_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Main M2 total loss is nonfinite")
        total_loss.backward()
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().norm().item() ** 2)
        grad_norm = squared_norm**0.5
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise FloatingPointError("Main M2 gradient norm is invalid")
        optimizer.step()
        batch_size = labels.numel()
        totals["ce"] += float(ce_loss.item()) * batch_size
        totals["axis"] += float(axis_loss.item()) * batch_size
        totals["total"] += float(total_loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += batch_size
        batches += 1
        grad_norm_sum += grad_norm
        if progress_every_batches and batches % progress_every_batches == 0:
            print(
                f"phase={phase} batch={batches} samples={samples} "
                f"ce={totals['ce'] / samples:.6f} "
                f"axis={totals['axis'] / samples:.6f} "
                f"total={totals['total'] / samples:.6f} "
                f"seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    if samples == 0:
        raise ValueError("Main M2 training loader produced no samples")
    end_cosine, end_angle = head_axis_geometry(model, frozen_axis)
    return {
        "ce_loss": totals["ce"] / samples,
        "axis_loss": totals["axis"] / samples,
        "total_loss": totals["total"] / samples,
        "accuracy": correct / samples,
        "mean_grad_norm": grad_norm_sum / batches,
        "samples": samples,
        "batches": batches,
        "fixed_axis_start_cosine": start_cosine,
        "fixed_axis_start_angle_degrees": start_angle,
        "fixed_axis_end_cosine": end_cosine,
        "fixed_axis_end_angle_degrees": end_angle,
    }


def _load_fold_rows(config: dict, fold: int) -> tuple[list, list, set, set]:
    assignments = read_fold_assignments(Path(config["fold_manifest"]))
    all_rows = load_window_rows([Path(config["index_path"])])
    validate_fold_assignments(
        assignments, expected_subjects={row.subject_id for row in all_rows}
    )
    training_subjects, validation_subjects = fold_subject_partitions(
        assignments, fold
    )
    train_rows = load_window_rows(
        [Path(config["index_path"])],
        include_subjects=training_subjects,
        max_windows_per_subject_per_class=int(
            config["training"]["max_windows_per_subject_per_class"]
        ),
        seed=int(config["seed"]),
    )
    validation_rows = load_window_rows(
        [Path(config["index_path"])], include_subjects=validation_subjects
    )
    return train_rows, validation_rows, training_subjects, validation_subjects


def _loaders(
    config: dict,
    train_rows: list[WindowRow],
    validation_rows: list[WindowRow],
) -> tuple[DataLoader, DataLoader, DataLoader, Any]:
    training = config["training"]
    sampler = build_subject_class_balanced_sampler(
        train_rows, seed=int(config["seed"]), num_samples=len(train_rows)
    )
    data_root = Path(config["data_root"])
    train_loader = DataLoader(
        ECGWindowDataset(train_rows, data_root=data_root),
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
    )
    axis_loader = DataLoader(
        ECGWindowDataset(train_rows, data_root=data_root),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
    )
    validation_loader = DataLoader(
        ECGWindowDataset(validation_rows, data_root=data_root),
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
    )
    return train_loader, axis_loader, validation_loader, sampler


def _assert_formal_git_clean(diagnostic: bool) -> dict:
    identity = git_identity()
    if not diagnostic and identity.get("dirty") is not False:
        raise ValueError("formal Main M2 training requires a clean Git worktree")
    return identity


def train_axis_fold(
    config: dict,
    *,
    fold: int,
    lambda_axis: float,
    device_request: str = "auto",
    output_override: Path | None = None,
    epoch_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    diagnostic_windows_per_class: int | None = None,
    resume: bool = False,
) -> dict:
    """Train one M2 lambda/fold with epoch-level source-axis refresh."""

    validate_m2_config(config)
    if fold not in range(5) or float(lambda_axis) not in map(float, config["lambdas"]):
        raise ValueError("unknown Main M2 fold or lambda")
    diagnostic = any(
        value is not None
        for value in (
            output_override,
            epoch_override,
            max_train_batches,
            max_eval_batches,
            diagnostic_windows_per_class,
        )
    )
    if diagnostic and output_override is None:
        raise ValueError("Main M2 diagnostics require an output override")
    git = _assert_formal_git_clean(diagnostic)
    output = output_override or (
        Path(config["output_dir"])
        / "folds"
        / lambda_slug(lambda_axis)
        / f"fold_{fold}"
    )
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Main M2 fold output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(int(config["seed"]))
    device = resolve_device(device_request)
    train_rows, validation_rows, training_subjects, validation_subjects = (
        _load_fold_rows(config, fold)
    )
    formal_train_count = len(train_rows)
    formal_validation_count = len(validation_rows)
    if diagnostic_windows_per_class is not None:
        train_rows = _balanced_diagnostic_rows(
            train_rows,
            per_class=int(diagnostic_windows_per_class),
            seed=int(config["seed"]),
        )
        validation_rows = _balanced_diagnostic_rows(
            validation_rows,
            per_class=int(diagnostic_windows_per_class),
            seed=int(config["seed"]),
        )
    train_loader, axis_loader, validation_loader, sampler = _loaders(
        config, train_rows, validation_rows
    )
    model = SourceMedTSTTT(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=int(config["training"]["patience"]))
    max_epochs = int(epoch_override or config["training"]["epochs"])
    progress = int(config["training"]["progress_every_batches"])
    provenance = {
        "experiment": config["experiment"],
        "formal": not diagnostic,
        "target_data_accessed": False,
        "fold": fold,
        "lambda_axis": float(lambda_axis),
        "seed": int(config["seed"]),
        "training_subjects": sorted(training_subjects),
        "validation_subjects": sorted(validation_subjects),
        "formal_training_windows": formal_train_count,
        "formal_validation_windows": formal_validation_count,
        "actual_training_windows": len(train_rows),
        "actual_validation_windows": len(validation_rows),
        "protocol_sha256": config["protocol_sha256"],
        "index_sha256": config["index_sha256"],
        "fold_manifest_sha256": config["fold_manifest_sha256"],
        "config": config,
        "git": git,
        "environment": environment_snapshot(device),
    }
    history: list[dict] = []
    start_epoch = 1
    initial_validation: dict | None = None
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError("Main M2 resume requires last.pt")
        payload = load_checkpoint(
            last_path, model=model, optimizer=optimizer, map_location=device
        )
        previous = payload.get("provenance", {})
        for key in ("fold", "lambda_axis", "protocol_sha256", "index_sha256"):
            if previous.get(key) != provenance.get(key):
                raise ValueError(f"Main M2 resume provenance mismatch: {key}")
        if previous.get("git", {}).get("commit") != git.get("commit"):
            raise ValueError("Main M2 resume requires the same code commit")
        sampler.generator.set_state(payload["sampler_generator_state"].cpu())
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        initial_validation = payload["initial_validation"]
        early_stopping.best = float(payload["best_validation_macro_f1"])
        early_stopping.bad_epochs = int(payload["early_stopping_bad_epochs"])
        provenance = previous
        provenance.setdefault("resume_events", []).append(
            {"from_epoch": int(payload["epoch"]), "git": git}
        )
    elif any(output.iterdir()):
        raise FileExistsError(f"Main M2 fold output is not empty: {output}")
    write_json(output / "run_manifest.json", provenance)

    if initial_validation is None:
        initial_validation = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_eval_batches,
            progress_every_batches=progress,
            phase="m2_initial_validation",
        )
    started = time.perf_counter()
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.perf_counter()
        axis, axis_stats = extract_source_axis(
            model,
            axis_loader,
            device=device,
            max_batches=max_eval_batches,
            progress_every_batches=progress,
            phase=f"m2_axis_lambda_{lambda_axis}_fold_{fold}_epoch_{epoch}",
        )
        train_metrics = train_axis_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            frozen_axis=axis,
            lambda_axis=float(lambda_axis),
            device=device,
            max_batches=max_train_batches,
            progress_every_batches=progress,
            phase=f"m2_train_lambda_{lambda_axis}_fold_{fold}_epoch_{epoch}",
        )
        validation = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
            max_batches=max_eval_batches,
            progress_every_batches=progress,
            phase=f"m2_validation_lambda_{lambda_axis}_fold_{fold}_epoch_{epoch}",
        )
        row = {
            "epoch": epoch,
            "axis_at_epoch_start": axis_stats,
            "train": train_metrics,
            "validation": validation,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        improved, should_stop = early_stopping.update(
            float(validation["macro_f1"])
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "sampler_generator_state": sampler.generator.get_state(),
            "best_validation_macro_f1": early_stopping.best,
            "early_stopping_bad_epochs": early_stopping.bad_epochs,
            "initial_validation": initial_validation,
            "history": history,
            "provenance": provenance,
        }
        save_checkpoint(last_path, checkpoint)
        if improved:
            save_checkpoint(best_path, checkpoint)
        write_json(output / "history.json", {"history": history})
        print(
            f"m2 lambda={lambda_axis:.2f} fold={fold} epoch={epoch} "
            f"ce={train_metrics['ce_loss']:.6f} "
            f"axis={train_metrics['axis_loss']:.6f} "
            f"val_f1={validation['macro_f1']:.6f} "
            f"angle={train_metrics['fixed_axis_end_angle_degrees']:.3f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )
        if not diagnostic and should_stop:
            break

    if not best_path.exists():
        raise RuntimeError("Main M2 training did not create a best checkpoint")
    best_payload = load_checkpoint(best_path, model=model, map_location=device)
    best_axis, best_axis_stats = extract_source_axis(
        model,
        axis_loader,
        device=device,
        max_batches=max_eval_batches,
        progress_every_batches=progress,
        phase=f"m2_best_axis_lambda_{lambda_axis}_fold_{fold}",
    )
    best_cosine, best_angle = head_axis_geometry(model, best_axis)
    best_validation = evaluate(
        model,
        validation_loader,
        criterion=criterion,
        device=device,
        max_batches=max_eval_batches,
        progress_every_batches=progress,
        phase=f"m2_best_validation_lambda_{lambda_axis}_fold_{fold}",
    )
    result = {
        "formal": not diagnostic,
        "target_data_accessed": False,
        "fold": fold,
        "lambda_axis": float(lambda_axis),
        "epochs_completed": len(history),
        "best_epoch": int(best_payload["epoch"]),
        "initial_validation": initial_validation,
        "best_validation": best_validation,
        "best_checkpoint_axis": best_axis_stats,
        "best_head_axis_cosine": best_cosine,
        "best_head_axis_angle_degrees": best_angle,
        "runtime_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "output_dir": str(output),
    }
    write_json(output / "result.json", result)
    return result


def train_axis_final(
    config: dict,
    *,
    device_request: str = "auto",
    output_override: Path | None = None,
    epoch_override: int | None = None,
    max_train_batches: int | None = None,
    max_axis_batches: int | None = None,
    diagnostic_windows_per_class: int | None = None,
    resume: bool = False,
) -> dict:
    """Train the fixed-epoch seed-42 full-source M2 model."""

    validate_m2_config(config)
    diagnostic = any(
        value is not None
        for value in (
            output_override,
            epoch_override,
            max_train_batches,
            max_axis_batches,
            diagnostic_windows_per_class,
        )
    )
    if diagnostic and output_override is None:
        raise ValueError("Main M2 final diagnostics require an output override")
    git = _assert_formal_git_clean(diagnostic)
    oof_dir = Path(config["output_dir"]) / "oof"
    selection_path = oof_dir / "selection_artifact.json"
    epoch_rule_path = oof_dir / "final_epoch_rule.json"
    selection = load_json(selection_path)
    epoch_rule = load_json(epoch_rule_path)
    if (
        not selection.get("frozen")
        or selection.get("status") != "SELECTED"
        or selection.get("target_data_accessed") is not False
        or (not diagnostic and not selection.get("formal"))
    ):
        raise ValueError("Main M2 final training requires a formal selected lambda")
    if (
        epoch_rule.get("rule") != config["full_source"]["epoch_rule"]
        or epoch_rule.get("selected_lambda") != selection.get("selected_lambda")
        or epoch_rule.get("selection_artifact_sha256")
        != sha256_file(selection_path)
        or epoch_rule.get("target_data_accessed") is not False
        or (not diagnostic and not epoch_rule.get("formal"))
    ):
        raise ValueError("Main M2 final epoch rule is invalid")
    formal_epochs = int(epoch_rule["final_epoch"])
    epochs = int(epoch_override or formal_epochs)
    if epochs <= 0:
        raise ValueError("Main M2 final epoch count must be positive")
    lambda_axis = float(selection["selected_lambda"])
    output = output_override or (
        Path(config["output_dir"])
        / "final_model"
        / f"seed_{int(config['full_source']['seed'])}"
    )
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Main M2 final output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seed = int(config["full_source"]["seed"])
    seed_everything(seed)
    device = resolve_device(device_request)
    assignments = read_fold_assignments(Path(config["fold_manifest"]))
    subjects = {row.subject_id for row in assignments}
    rows = load_window_rows(
        [Path(config["index_path"])],
        include_subjects=subjects,
        max_windows_per_subject_per_class=int(
            config["training"]["max_windows_per_subject_per_class"]
        ),
        seed=seed,
    )
    formal_training_windows = len(rows)
    if formal_training_windows != int(
        config["full_source"]["expected_training_windows"]
    ):
        raise ValueError("Main M2 full-source training count changed")
    if diagnostic_windows_per_class is not None:
        rows = _balanced_diagnostic_rows(
            rows,
            per_class=int(diagnostic_windows_per_class),
            seed=seed,
        )
    training = config["training"]
    sampler = build_subject_class_balanced_sampler(
        rows, seed=seed, num_samples=len(rows)
    )
    dataset = ECGWindowDataset(rows, data_root=Path(config["data_root"]))
    train_loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
    )
    axis_loader = DataLoader(
        dataset,
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=False,
    )
    model = SourceMedTSTTT(**config["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    provenance = {
        "experiment": "main_m2_axis_alignment_final",
        "formal": not diagnostic,
        "target_data_accessed": False,
        "seed": seed,
        "lambda_axis": lambda_axis,
        "epochs": epochs,
        "formal_epochs": formal_epochs,
        "subjects": sorted(subjects),
        "subject_count": len(subjects),
        "formal_training_windows": formal_training_windows,
        "actual_training_windows": len(rows),
        "selection_artifact_sha256": sha256_file(selection_path),
        "final_epoch_rule_sha256": sha256_file(epoch_rule_path),
        "protocol_sha256": config["protocol_sha256"],
        "index_sha256": config["index_sha256"],
        "fold_manifest_sha256": config["fold_manifest_sha256"],
        "selected_oof_threshold": selection["selected_threshold"],
        "config": config,
        "git": git,
        "environment": environment_snapshot(device),
    }
    checkpoint_path = output / "final.pt"
    history: list[dict] = []
    start_epoch = 1
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError("Main M2 final resume requires final.pt")
        payload = load_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, map_location=device
        )
        previous = payload.get("provenance", {})
        for key in (
            "lambda_axis",
            "protocol_sha256",
            "selection_artifact_sha256",
            "final_epoch_rule_sha256",
        ):
            if previous.get(key) != provenance.get(key):
                raise ValueError(f"Main M2 final resume provenance mismatch: {key}")
        if previous.get("git", {}).get("commit") != git.get("commit"):
            raise ValueError("Main M2 final resume requires the same code commit")
        sampler.generator.set_state(payload["sampler_generator_state"].cpu())
        history = list(payload["history"])
        start_epoch = int(payload["epoch"]) + 1
        provenance = previous
        provenance.setdefault("resume_events", []).append(
            {"from_epoch": int(payload["epoch"]), "git": git}
        )
    elif any(output.iterdir()):
        raise FileExistsError(f"Main M2 final output is not empty: {output}")
    write_json(output / "run_manifest.json", provenance)

    started = time.perf_counter()
    progress = int(training["progress_every_batches"])
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.perf_counter()
        axis, axis_stats = extract_source_axis(
            model,
            axis_loader,
            device=device,
            max_batches=max_axis_batches,
            progress_every_batches=progress,
            phase=f"m2_final_axis_epoch_{epoch}",
        )
        train_metrics = train_axis_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            frozen_axis=axis,
            lambda_axis=lambda_axis,
            device=device,
            max_batches=max_train_batches,
            progress_every_batches=progress,
            phase=f"m2_final_train_epoch_{epoch}",
        )
        row = {
            "epoch": epoch,
            "axis_at_epoch_start": axis_stats,
            "train": train_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        save_checkpoint(
            checkpoint_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_generator_state": sampler.generator.get_state(),
                "history": history,
                "provenance": provenance,
            },
        )
        write_json(output / "history.json", {"history": history})
        print(
            f"m2 final epoch={epoch}/{epochs} ce={train_metrics['ce_loss']:.6f} "
            f"axis={train_metrics['axis_loss']:.6f} "
            f"angle={train_metrics['fixed_axis_end_angle_degrees']:.3f} "
            f"seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )

    final_axis, final_axis_stats = extract_source_axis(
        model,
        axis_loader,
        device=device,
        max_batches=max_axis_batches,
        progress_every_batches=progress,
        phase="m2_final_axis_freeze",
    )
    final_cosine, final_angle = head_axis_geometry(model, final_axis)
    save_checkpoint(
        checkpoint_path,
        {
            "epoch": epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "sampler_generator_state": sampler.generator.get_state(),
            "history": history,
            "final_source_axis": final_axis.cpu(),
            "final_source_axis_stats": final_axis_stats,
            "final_head_axis_cosine": final_cosine,
            "final_head_axis_angle_degrees": final_angle,
            "provenance": provenance,
        },
    )
    result = {
        "formal": not diagnostic,
        "target_data_accessed": False,
        "seed": seed,
        "lambda_axis": lambda_axis,
        "epochs_completed": epochs,
        "formal_training_windows": formal_training_windows,
        "actual_training_windows": len(rows),
        "final_train": history[-1]["train"],
        "final_source_axis_stats": final_axis_stats,
        "final_head_axis_cosine": final_cosine,
        "final_head_axis_angle_degrees": final_angle,
        "selected_oof_threshold": selection["selected_threshold"],
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output_dir": str(output),
    }
    write_json(output / "result.json", result)
    return result
