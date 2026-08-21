import numpy as np
import pytest

from src.evaluation.main_three_targets import (
    decision_status,
    mechanism_statistics,
    write_score_archive,
)


def _valid_arrays():
    return {
        "dataset": np.asarray(["x", "x"]),
        "subject_id": np.asarray(["a", "b"]),
        "record_id": np.asarray(["r1", "r2"]),
        "window_start": np.asarray([0, 1], dtype=np.int64),
        "raw_logit_difference": np.asarray([-1.0, 1.0], dtype=np.float32),
    }


def test_target_archive_is_label_free_and_exact_schema(tmp_path):
    arrays = _valid_arrays()
    arrays["labels"] = np.asarray([0, 1])
    with pytest.raises(ValueError, match="cannot contain labels"):
        write_score_archive(tmp_path / "scores.npz", arrays)


def test_target_archive_rejects_nonfinite_scores(tmp_path):
    arrays = _valid_arrays()
    arrays["raw_logit_difference"][1] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        write_score_archive(tmp_path / "scores.npz", arrays)


def test_target_archive_rejects_misaligned_arrays(tmp_path):
    arrays = _valid_arrays()
    arrays["record_id"] = np.asarray(["r1"])
    with pytest.raises(ValueError, match="misaligned"):
        write_score_archive(tmp_path / "scores.npz", arrays)


def test_mechanism_statistics_gap_dprime_overlap():
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([-2.0, -1.0, 1.0, 2.0])
    result = mechanism_statistics(labels, scores, bins=4)
    assert result["class_gap"] == 3.0
    assert np.isclose(result["d_prime"], 6.0)
    assert result["histogram_overlap_coefficient"] == 0.0
    assert result["score_range_rule"] == "target_unlabeled_observed_min_max"


def test_decision_requires_both_ranking_and_two_operating_improvements():
    reference = {key: 0.5 for key in ("auroc", "auprc", "balanced_accuracy", "macro_f1", "mcc")}
    current = {
        "auroc": 0.6,
        "auprc": 0.6,
        "balanced_accuracy": 0.6,
        "macro_f1": 0.6,
        "mcc": 0.4,
    }
    assert decision_status(current, reference)["status"] == "strong_success_pending_target_consistency"


def test_decision_failure_requires_ranking_and_operating_decline():
    reference = {key: 0.5 for key in ("auroc", "auprc", "balanced_accuracy", "macro_f1", "mcc")}
    current = {
        "auroc": 0.4,
        "auprc": 0.4,
        "balanced_accuracy": 0.4,
        "macro_f1": 0.4,
        "mcc": 0.6,
    }
    assert decision_status(current, reference)["status"] == "failure"
