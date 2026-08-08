"""Validation macro-F1 early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: float = float("-inf")
    bad_epochs: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
