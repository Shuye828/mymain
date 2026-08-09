import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.target_gmm_evaluation import evaluate_frozen_target_gmm
from src.training.reproducibility import sha256_file


def _write_target_index(path: Path) -> None:
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
        for split_index, split in enumerate(("adaptation", "evaluation")):
            for index, label in enumerate((0, 0, 1, 1)):
                writer.writerow(
                    {
                        "dataset": "afdb",
                        "record_id": f"record_{split_index}_{index}",
                        "subject_id": f"subject_{split_index}",
                        "start_sample": index * 2500,
                        "end_sample": (index + 1) * 2500,
                        "fs_original": 250,
                        "binary_label": label,
                        "rhythm_label": "fixture",
                        "source_split": "test",
                        "target_split": split,
                    }
                )


def _write_frozen_fixture(fit_dir: Path, *, leak_labels: bool = False) -> None:
    fit_dir.mkdir()
    index_path = fit_dir / "target.csv"
    _write_target_index(index_path)
    subjects = []
    records = []
    starts = []
    splits = []
    probabilities = []
    for split_index, split in enumerate(("adaptation", "evaluation")):
        for index, label in enumerate((0, 0, 1, 1)):
            subjects.append(f"subject_{split_index}")
            records.append(f"record_{split_index}_{index}")
            starts.append(index * 2500)
            splits.append(split)
            probabilities.append(0.1 if label == 0 else 0.9)
    arrays = {
        "dataset": np.asarray(["afdb"] * 8),
        "subject_id": np.asarray(subjects),
        "record_id": np.asarray(records),
        "window_start": np.asarray(starts, dtype=np.int64),
        "target_split": np.asarray(splits),
        "source_classifier_probability": np.asarray(probabilities),
        "direction_score": np.asarray([-1, -0.8, 0.8, 1] * 2),
        "inductive_gmm_af_probability": np.asarray(probabilities),
        "transductive_gmm_af_probability": np.asarray(probabilities),
    }
    if leak_labels:
        arrays["labels"] = np.asarray([0, 0, 1, 1] * 2)
    score_path = fit_dir / "target_scores.npz"
    np.savez_compressed(score_path, **arrays)
    gmm = {
        "frozen": True,
        "labels_accessed": False,
        "source_dataset": "ltafdb",
        "target_dataset": "afdb",
        "target_index_sha256": sha256_file(index_path),
        "target_score_sha256": sha256_file(score_path),
        "protocols": {
            name: {
                "gmm": {
                    "reliable": True,
                    "reliability_failures": [],
                    "source_fixed_threshold": 0.0,
                }
            }
            for name in ("inductive_holdout", "transductive")
        },
    }
    gmm_path = fit_dir / "gmm_artifact.json"
    gmm_path.write_text(json.dumps(gmm), encoding="utf-8")
    manifest = {
        "git": {"commit": "fixture", "dirty": False},
        "config": {"target_index": str(index_path)},
        "labels_accessed": False,
        "diagnostic_max_batches": None,
        "gmm_artifact_sha256": sha256_file(gmm_path),
    }
    (fit_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_frozen_target_evaluation_joins_labels_after_freeze(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fit"
    _write_frozen_fixture(fit_dir)

    result = evaluate_frozen_target_gmm(fit_dir)

    assert result["adaptation_labels_accessed"] is False
    assert result["evaluation_labels_accessed_after_freeze"] is True
    assert result["protocols"]["inductive_holdout"]["support"] == 4
    assert result["protocols"]["transductive"]["support"] == 8
    assert (
        result["protocols"]["inductive_holdout"]["B4_target_gmm"][
            "macro_f1"
        ]
        == 1.0
    )


def test_frozen_target_evaluation_rejects_label_leakage(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fit"
    _write_frozen_fixture(fit_dir, leak_labels=True)

    with pytest.raises(ValueError, match="leaked labels"):
        evaluate_frozen_target_gmm(fit_dir)
