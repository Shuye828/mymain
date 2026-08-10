import csv
from pathlib import Path

import numpy as np
import pytest

from src.analysis.axis_distribution_shift import (
    _write_score_archive,
    boundary_shift_statistics,
    distribution_statistics,
    load_hidden_selected_rows,
)


def test_hidden_selected_loader_never_parses_label_fields(tmp_path: Path) -> None:
    path = tmp_path / "selected.csv"
    fields = [
        "dataset",
        "record_id",
        "subject_id",
        "start_sample",
        "end_sample",
        "fs_original",
        "binary_label",
        "rhythm_label",
        "source_split",
        "target_split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "fixture",
                "record_id": "r",
                "subject_id": "s",
                "start_sample": 0,
                "end_sample": 2000,
                "fs_original": 200,
                "binary_label": "must_not_parse",
                "rhythm_label": "must_not_parse",
                "source_split": "train",
                "target_split": "evaluation",
            }
        )
    rows = load_hidden_selected_rows(path)
    assert rows[0].binary_label == -1
    assert rows[0].rhythm_label == "__HIDDEN_STAGE5D_LABEL__"


def test_distribution_statistics_compute_gap_dprime_and_overlap() -> None:
    result = distribution_statistics(
        np.array([0, 0, 1, 1]),
        np.array([-0.8, -0.6, 0.6, 0.8]),
        bins=20,
    )
    assert np.isclose(result["mu_nonaf"], -0.7)
    assert np.isclose(result["mu_af"], 0.7)
    assert np.isclose(result["class_gap"], 1.4)
    assert np.isclose(result["d_prime"], 14.0)
    assert result["histogram_overlap_coefficient"] == 0.0
    assert result["auroc"] == 1.0


def test_overlap_is_one_for_identical_empirical_distributions() -> None:
    result = distribution_statistics(
        np.array([0, 0, 1, 1]),
        np.array([-0.5, 0.5, -0.5, 0.5]),
        bins=20,
    )
    assert result["histogram_overlap_coefficient"] == 1.0
    assert result["d_prime"] == 0.0


def test_boundary_shift_uses_source_midpoint_gap_and_p1() -> None:
    shift = boundary_shift_statistics(
        {"midpoint": 0.3, "class_gap": 1.0},
        {"midpoint": 0.1, "class_gap": 2.0},
        p0_threshold=0.0,
        p1_threshold=0.2,
        oracle_threshold=0.5,
    )
    assert np.isclose(shift["oracle_minus_p1"], 0.3)
    assert np.isclose(shift["oracle_minus_p0"], 0.5)
    assert np.isclose(shift["midpoint_drift"], 0.2)
    assert np.isclose(shift["class_gap_ratio"], 0.5)


def test_score_archive_rejects_label_leakage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot contain labels"):
        _write_score_archive(
            tmp_path / "scores.npz",
            {"axis_score": np.array([0.1]), "binary_label": np.array([1])},
        )
