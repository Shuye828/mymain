import csv

import numpy as np
import pytest

from src.data.ecg_dataset import WindowRow
from src.training.afdb_oof_finalize import row_identity, validate_oof_coverage
from src.training.afdb_full_source import resolve_final_epoch
from src.training.afdb_source_protocol import (
    build_fold_assignments,
    centered_prototype_margin,
    fold_subject_partitions,
    median_best_epoch,
    validate_fold_assignments,
)


def test_fold_assignment_is_deterministic_complete_and_exclusive() -> None:
    subjects = [f"s{index:02d}" for index in range(23)]
    first = build_fold_assignments(subjects)
    second = build_fold_assignments(list(reversed(subjects)))

    assert first == second
    assert sorted(sum(([row.subject_id] for row in first), [])) == sorted(subjects)
    assert sorted(
        sum(([sum(row.fold_id == fold for row in first)] for fold in range(5)), [])
    ) == [4, 4, 5, 5, 5]
    validate_fold_assignments(first, expected_subjects=set(subjects))


def test_fold_validation_rejects_duplicate_subject() -> None:
    rows = build_fold_assignments([f"s{index:02d}" for index in range(23)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_fold_assignments(
            rows + [rows[0]], expected_subjects={r.subject_id for r in rows}
        )


def test_fold_subject_partitions_are_disjoint_and_complete() -> None:
    rows = build_fold_assignments([f"s{index:02d}" for index in range(23)])
    training, validation = fold_subject_partitions(rows, 0)
    assert not training & validation
    assert training | validation == {row.subject_id for row in rows}
    assert len(training) == 18
    assert len(validation) == 5


def test_median_best_epoch_requires_exactly_five_positive_epochs() -> None:
    assert median_best_epoch([9, 3, 7, 5, 11]) == 7
    with pytest.raises(ValueError):
        median_best_epoch([1, 2, 3, 4])


def test_centered_prototype_margin_has_zero_class_midpoint() -> None:
    features = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    direction, margins, midpoint = centered_prototype_margin(features, labels)

    assert np.allclose(direction, [1.0, 0.0])
    assert midpoint == 0.0
    assert np.isclose(
        (margins[labels == 0].mean() + margins[labels == 1].mean()) / 2, 0
    )


def test_explicit_subject_loader_filter(tmp_path) -> None:
    from src.data.ecg_dataset import load_window_rows

    path = tmp_path / "index.csv"
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
        for index, subject in enumerate(("a", "b", "a")):
            writer.writerow(
                {
                    "dataset": "afdb",
                    "record_id": subject,
                    "subject_id": subject,
                    "start_sample": index * 2500,
                    "end_sample": (index + 1) * 2500,
                    "fs_original": 250,
                    "binary_label": index % 2,
                    "rhythm_label": "fixture",
                    "source_split": "historical",
                    "target_split": "evaluation",
                }
            )
    rows = load_window_rows([path], include_subjects={"a"})
    assert len(rows) == 2
    assert {row.subject_id for row in rows} == {"a"}


def test_oof_coverage_requires_unique_exact_window_identity() -> None:
    rows = [
        WindowRow("afdb", "r1", "s1", 0, 2500, 250, 0, "N", "x", "x"),
        WindowRow("afdb", "r2", "s2", 0, 2500, 250, 1, "AF", "x", "x"),
    ]
    identities = [row_identity(row) for row in rows]
    validate_oof_coverage(rows, identities)
    with pytest.raises(ValueError, match="duplicate"):
        validate_oof_coverage(rows, [identities[0], identities[0]])
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_oof_coverage(rows, identities[:1])


def test_final_epoch_rule_must_be_frozen_and_source_only() -> None:
    config = {"full_source": {"epoch_rule": "median"}}
    rule = {
        "frozen": True,
        "target_data_accessed": False,
        "rule": "median",
        "final_epoch": 7,
        "best_epochs": [3, 5, 7, 9, 11],
    }
    assert resolve_final_epoch(config, rule) == 7
    with pytest.raises(ValueError, match="source-only"):
        resolve_final_epoch(config, {**rule, "target_data_accessed": True})
