import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.source_threshold_baseline import (
    select_source_thresholds,
    threshold_curve,
)
from src.training.reproducibility import sha256_file


def test_threshold_curve_finds_perfect_balanced_accuracy() -> None:
    rows, selected = threshold_curve(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        fixed_threshold=0.0,
    )
    assert len(rows) == 5
    assert selected["threshold"] == 0.8
    assert selected["balanced_accuracy"] == 1.0


def test_threshold_tie_break_uses_macro_f1_then_distance_then_lower() -> None:
    _, selected = threshold_curve(
        np.array([1, 0, 1, 0]),
        np.array([9.0, 8.0, 7.0, 6.0]),
        fixed_threshold=8.0,
    )
    assert selected["balanced_accuracy"] == 0.75
    assert selected["threshold"] == 7.0
    assert selected["optimal_balanced_accuracy_tie_count"] == 2


def test_threshold_change_preserves_ranking_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([-2.0, -1.0, 0.5, 4.0])
    fixed = compute_binary_metrics(labels, scores, threshold=0.0)
    optimized = compute_binary_metrics(labels, scores, threshold=0.5)
    assert fixed["auroc"] == optimized["auroc"]
    assert fixed["auprc"] == optimized["auprc"]


def _selection_fixture(tmp_path: Path) -> dict:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("stage5c\n", encoding="utf-8")
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
    source_index = tmp_path / "source.csv"
    rows = []
    with source_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate((0, 0, 1, 1)):
            row = {
                "dataset": "fixture",
                "record_id": f"r{index}",
                "subject_id": f"s{index}",
                "start_sample": index * 2000,
                "end_sample": (index + 1) * 2000,
                "fs_original": 200,
                "binary_label": label,
                "rhythm_label": "fixture",
                "source_split": "validation",
                "target_split": "evaluation",
            }
            writer.writerow(row)
            rows.append(row)
    source_config = tmp_path / "source_config.json"
    source_config.write_text(
        json.dumps(
            {"role": "source", "dataset": "fixture", "index_path": str(source_index)}
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    direction = tmp_path / "direction.json"
    direction.write_text(json.dumps({"source_fixed_threshold": 0.25}), encoding="utf-8")
    score_dir = tmp_path / "stage5a"
    score_dir.mkdir()
    score_path = score_dir / "scores.npz"
    np.savez_compressed(
        score_path,
        dataset=np.array(["fixture"] * 4),
        subject_id=np.array([row["subject_id"] for row in rows]),
        record_id=np.array([row["record_id"] for row in rows]),
        window_start=np.array([row["start_sample"] for row in rows]),
        analysis_scope=np.array(["source_validation"] * 4),
        classifier_logit_difference=np.array([-2.0, -1.0, 1.0, 2.0]),
        prototype_score=np.array([-0.8, -0.5, 0.7, 0.9]),
    )
    score_artifact = score_dir / "score_artifact.json"
    score_artifact.write_text(
        json.dumps(
            {
                "frozen": True,
                "target_labels_accessed": False,
                "diagnostic_max_batches_per_split": None,
                "source_dataset": "fixture",
                "target_dataset": "missing_target",
                "source_index_sha256": sha256_file(source_index),
                "target_index_sha256": "frozen-target-hash",
                "checkpoint_sha256": sha256_file(checkpoint),
                "direction_sha256": sha256_file(direction),
                "score_sha256": sha256_file(score_path),
            }
        ),
        encoding="utf-8",
    )
    (score_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "target_labels_accessed": False,
                "config": {"experiment": "stage5a_head_vs_direction"},
                "score_artifact_sha256": sha256_file(score_artifact),
            }
        ),
        encoding="utf-8",
    )
    return {
        "experiment": "fixture",
        "role": "source_threshold_baseline",
        "protocol": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "threshold_selection_label_scope": "source_validation_only",
        "adaptation_time_target_label_access": "prohibited",
        "sources": {
            "fixture": {
                "source_config": str(source_config),
                "checkpoint": str(checkpoint),
                "direction": str(direction),
                "stage5a_score_dir": str(score_dir),
                "target_dataset": "missing_target",
                "target_index": str(tmp_path / "this_target_must_not_be_opened.csv"),
            }
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_selection_does_not_open_target_index_or_access_target_labels(
    tmp_path: Path,
) -> None:
    config = _selection_fixture(tmp_path)
    result = select_source_thresholds(config, source_name="fixture")
    manifest = json.loads(
        (Path(result["output_dir"]) / "selection_manifest.json").read_text()
    )
    assert result["target_labels_accessed"] is False
    assert manifest["target_index_opened"] is False
    assert manifest["target_labels_accessed"] is False
    assert result["thresholds"]["H0"] == 0.0
    assert result["thresholds"]["P0"] == 0.25
