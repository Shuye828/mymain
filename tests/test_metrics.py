import numpy as np

from src.evaluation.metrics import compute_binary_metrics


def test_binary_metrics_perfect_predictions() -> None:
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "precision",
        "mcc",
    ):
        assert np.isclose(metrics[key], 1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
