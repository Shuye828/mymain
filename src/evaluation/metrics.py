"""Binary window-level metrics required by the experiment protocol."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


def _safe_metric(function, *args, **kwargs) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return math.nan


def compute_binary_metrics(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """Compute frozen-threshold metrics without choosing on target labels."""

    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("labels and probabilities must be aligned 1-D arrays")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty dataset")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")
    if not np.isfinite(y_prob).all():
        raise ValueError("probabilities contain NaN/Inf")
    y_pred = (y_prob >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )
    has_both_classes = bool(np.unique(y_true).size == 2)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob)
        if has_both_classes
        else math.nan,
        "auprc": _safe_metric(average_precision_score, y_true, y_prob)
        if has_both_classes
        else math.nan,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if has_both_classes
        else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "negative_count": int((y_true == 0).sum()),
    }
