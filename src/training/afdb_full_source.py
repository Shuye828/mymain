"""Fixed-epoch full-source AFDB training for Revision R2."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    ECGWindowDataset,
    build_subject_class_balanced_sampler,
    load_window_rows,
)
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.afdb_source_protocol import (
    config_protocol_hash,
    read_fold_assignments,
    write_json,
)
from src.training.checkpointing import save_checkpoint
from src.training.engine import train_one_epoch
from src.training.reproducibility import (
    environment_snapshot,
    git_identity,
    resolve_device,
    seed_everything,
    sha256_file,
)


def resolve_final_epoch(config: dict, rule: dict) -> int:
    if not rule.get("frozen") or rule.get("target_data_accessed") is not False:
        raise ValueError(
            "full-source training requires a frozen source-only epoch rule"
        )
    if rule.get("rule") != config["full_source"]["epoch_rule"]:
        raise ValueError("final epoch aggregation rule mismatch")
    epoch = int(rule.get("final_epoch", 0))
    if epoch <= 0 or len(rule.get("best_epochs", [])) != 5:
        raise ValueError("final epoch artifact is incomplete")
    return epoch


def train_afdb_full_source(
    config: dict,
    *,
    seed: int,
    device_request: str = "auto",
    output_override: Path | None = None,
    epoch_override: int | None = None,
    max_train_batches: int | None = None,
) -> dict:
    config_protocol_hash(config)
    allowed_seeds = [int(value) for value in config["full_source"]["seeds"]]
    if seed not in allowed_seeds:
        raise ValueError(f"seed {seed} is not frozen for R2 full-source training")
    if (
        epoch_override is not None or max_train_batches is not None
    ) and output_override is None:
        raise ValueError("full-source diagnostics require an explicit output override")
    rule_path = Path(config["output_dir"]) / "final_epoch_rule.json"
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    formal_epoch = resolve_final_epoch(config, rule)
    epochs = int(epoch_override if epoch_override is not None else formal_epoch)
    if epochs <= 0:
        raise ValueError("full-source epoch count must be positive")

    seed_everything(seed)
    device = resolve_device(device_request)
    assignments = read_fold_assignments(Path(config["fold_manifest"]))
    subjects = {row.subject_id for row in assignments}
    training = config["training"]
    rows = load_window_rows(
        [Path(config["index_path"])],
        include_subjects=subjects,
        max_windows_per_subject_per_class=int(
            training["max_windows_per_subject_per_class"]
        ),
        seed=seed,
    )
    dataset = ECGWindowDataset(rows, data_root=Path(config["data_root"]))
    sampler = build_subject_class_balanced_sampler(
        rows, seed=seed, num_samples=len(rows)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        sampler=sampler,
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
    output = output_override or (
        Path(config["output_dir"]) / "full_source" / f"seed_{seed}"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"full-source output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    diagnostic = output_override is not None
    provenance = {
        "experiment": "revision_r2_afdb_full_source",
        "formal": not diagnostic,
        "seed": seed,
        "main_seed": seed == 42,
        "epochs": epochs,
        "formal_epoch": formal_epoch,
        "target_data_accessed": False,
        "subjects": sorted(subjects),
        "subject_count": len(subjects),
        "training_windows": len(rows),
        "index_sha256": sha256_file(Path(config["index_path"])),
        "fold_manifest_sha256": sha256_file(Path(config["fold_manifest"])),
        "final_epoch_rule_sha256": sha256_file(rule_path),
        "config": config,
        "git": git_identity(),
        "environment": environment_snapshot(device),
    }
    write_json(output / "run_manifest.json", provenance)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        metrics = train_one_epoch(
            model,
            loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_train_batches,
            progress_every_batches=int(training["progress_every_batches"]),
            phase=f"full_source_seed_{seed}_epoch_{epoch}",
        )
        row = {
            "epoch": epoch,
            "train": metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        save_checkpoint(
            output / "final.pt",
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
                "provenance": provenance,
            },
        )
        write_json(output / "history.json", {"history": history})
        print(
            f"epoch={epoch}/{epochs} train_loss={metrics['loss']:.6f} "
            f"train_acc={metrics['accuracy']:.4f} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
    result = {
        "formal": not diagnostic,
        "seed": seed,
        "epochs_completed": epochs,
        "training_windows": len(rows),
        "subject_count": len(subjects),
        "final_train": history[-1]["train"],
        "runtime_seconds": time.perf_counter() - started,
        "output_dir": str(output),
        "checkpoint_sha256": sha256_file(output / "final.pt"),
    }
    write_json(output / "result.json", result)
    return result
