"""Reusable supervised epoch loops for source-only classification."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import compute_binary_metrics


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    progress_every_batches: int | None = None,
    phase: str = "train",
) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    samples = 0
    grad_norm_sum = 0.0
    batches = 0
    progress_start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        loss.backward()
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += float(parameter.grad.detach().norm().item() ** 2)
        grad_norm = squared_norm**0.5
        if not np.isfinite(grad_norm) or grad_norm == 0:
            raise FloatingPointError(f"invalid gradient norm: {grad_norm}")
        optimizer.step()
        batch_size = labels.numel()
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += batch_size
        grad_norm_sum += grad_norm
        batches += 1
        if progress_every_batches and batches % progress_every_batches == 0:
            print(
                f"phase={phase} batch={batches} samples={samples} "
                f"loss={loss_sum / samples:.6f} "
                f"seconds={time.perf_counter() - progress_start:.1f}",
                flush=True,
            )
    if samples == 0:
        raise ValueError("training loader produced no samples")
    return {
        "loss": loss_sum / samples,
        "accuracy": correct / samples,
        "mean_grad_norm": grad_norm_sum / batches,
        "samples": float(samples),
        "batches": float(batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    *,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    progress_every_batches: int | None = None,
    phase: str = "evaluation",
) -> dict:
    model.eval()
    loss_sum = 0.0
    samples = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    batches = 0
    progress_start = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["x"].to(device)
        labels = batch["y"].to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        batch_size = labels.numel()
        loss_sum += float(loss.item()) * batch_size
        samples += batch_size
        labels_all.append(labels.cpu().numpy())
        probabilities_all.append(probabilities.cpu().numpy())
        batches += 1
        if progress_every_batches and batches % progress_every_batches == 0:
            print(
                f"phase={phase} batch={batches} samples={samples} "
                f"loss={loss_sum / samples:.6f} "
                f"seconds={time.perf_counter() - progress_start:.1f}",
                flush=True,
            )
    if samples == 0:
        raise ValueError("evaluation loader produced no samples")
    metrics = compute_binary_metrics(
        np.concatenate(labels_all), np.concatenate(probabilities_all)
    )
    metrics["loss"] = loss_sum / samples
    return metrics
