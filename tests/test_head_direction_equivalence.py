import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.analysis.head_direction_equivalence import (
    _equivalence_status,
    _write_score_archive,
    compare_directions,
    compute_ranking_metrics,
    compute_score_correlations,
    compute_stage5a_scores,
    finalize_head_direction_analysis,
)
from src.training.reproducibility import sha256_file


def test_direction_comparison_uses_af_minus_nonaf_orientation() -> None:
    comparison = compare_directions(
        np.array([1.0, 0.0]),
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        np.array([-0.25, 0.75]),
    )

    assert np.isclose(comparison["cosine"], 1.0)
    assert np.isclose(comparison["angle_degrees"], 0.0)
    assert np.isclose(comparison["head_bias_difference"], 1.0)
    assert comparison["head_direction"] == [1.0, 0.0]


def test_stage5a_scores_use_normalized_feature_and_exact_logits() -> None:
    logits = torch.tensor([[4.0, 1.5], [-2.0, 3.0]])
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    direction = torch.tensor([1.0, 0.0])

    prototype, classifier = compute_stage5a_scores(
        logits, features, direction
    )

    assert torch.allclose(prototype, torch.tensor([0.6, 0.0]))
    assert torch.equal(classifier, torch.tensor([-2.5, 5.0]))


def test_correlations_are_computed_separately_for_all_scopes() -> None:
    scopes = np.repeat(
        np.asarray(
            ["source_validation", "source_test", "target_evaluation"]
        ),
        4,
    )
    prototype = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 3)
    classifier = 3.0 * prototype + 0.5

    correlations = compute_score_correlations(
        scopes, prototype, classifier
    )

    assert set(correlations) == {
        "source_validation",
        "source_test",
        "target_evaluation",
    }
    assert all(
        np.isclose(payload["pearson"], 1.0)
        and np.isclose(payload["spearman"], 1.0)
        for payload in correlations.values()
    )


def test_ranking_metrics_use_continuous_score_without_threshold() -> None:
    metrics = compute_ranking_metrics(
        np.array([0, 0, 1, 1]), np.array([-2.0, -1.0, 0.5, 4.0])
    )

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["support"] == 4


def test_diagnostic_ranking_records_single_class_as_unavailable() -> None:
    metrics = compute_ranking_metrics(
        np.array([0, 0]),
        np.array([-2.0, -1.0]),
        allow_single_class=True,
    )

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    with pytest.raises(ValueError, match="both binary classes"):
        compute_ranking_metrics(np.array([0, 0]), np.array([-2.0, -1.0]))


def test_equivalence_rule_uses_worst_split_spearman() -> None:
    correlations = {
        "source_validation": {"spearman": 0.999},
        "source_test": {"spearman": 0.997},
        "target_evaluation": {"spearman": 0.979},
    }

    status = _equivalence_status(
        0.99,
        correlations,
        {"min_direction_cosine": 0.95, "min_split_spearman": 0.98},
    )

    assert status["conclusion"] == "partially_equivalent"
    assert status["uses_labels"] is False
    assert np.isclose(status["minimum_split_spearman"], 0.979)


def test_score_archive_rejects_any_label_field(tmp_path: Path) -> None:
    arrays = {
        "prototype_score": np.array([0.1, 0.2]),
        "classifier_logit_difference": np.array([0.3, 0.4]),
        "labels": np.array([0, 1]),
    }

    with pytest.raises(ValueError, match="cannot contain labels"):
        _write_score_archive(tmp_path / "scores.npz", arrays)


def _write_index(path: Path, dataset: str, split_field: str) -> list[dict]:
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
    rows: list[dict] = []
    splits = (
        ("validation", "test")
        if split_field == "source"
        else ("evaluation",)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_index, split in enumerate(splits):
            for index, label in enumerate((0, 0, 1, 1)):
                row = {
                    "dataset": dataset,
                    "record_id": f"record_{split_index}_{index}",
                    "subject_id": f"subject_{split_index}",
                    "start_sample": index * 2000,
                    "end_sample": (index + 1) * 2000,
                    "fs_original": 200,
                    "binary_label": label,
                    "rhythm_label": "fixture",
                    "source_split": split if split_field == "source" else "test",
                    "target_split": (
                        "adaptation"
                        if split_field == "source"
                        else "evaluation"
                    ),
                }
                writer.writerow(row)
                rows.append(row)
    return rows


def _write_finalize_fixture(tmp_path: Path, *, leak_labels: bool = False) -> tuple[dict, Path]:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    source_index = tmp_path / "source.csv"
    target_index = tmp_path / "target.csv"
    source_rows = _write_index(source_index, "source_fixture", "source")
    target_rows = _write_index(target_index, "target_fixture", "target")
    source_config = {
        "role": "source",
        "dataset": "source_fixture",
        "index_path": str(source_index),
        "data_root": str(tmp_path),
        "model": {"dim": 2},
    }
    source_config_path = tmp_path / "source_config.json"
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"fixture checkpoint")
    direction_dir = tmp_path / "direction"
    direction_dir.mkdir()
    direction_path = direction_dir / "disease_direction.json"
    direction_path.write_text(
        json.dumps(
            {
                "dataset": "source_fixture",
                "representation": {"kind": "backbone_l2"},
                "direction": [1.0, 0.0],
            }
        ),
        encoding="utf-8",
    )
    (direction_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "index_sha256": sha256_file(source_index),
                "diagnostic_max_batches": None,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    rows_by_scope = {
        "source_validation": source_rows[:4],
        "source_test": source_rows[4:],
        "target_evaluation": target_rows,
    }
    datasets = []
    subjects = []
    records = []
    starts = []
    scopes = []
    prototype = []
    classifier = []
    labels = []
    for scope, rows in rows_by_scope.items():
        for row in rows:
            label = int(row["binary_label"])
            datasets.append(row["dataset"])
            subjects.append(row["subject_id"])
            records.append(row["record_id"])
            starts.append(int(row["start_sample"]))
            scopes.append(scope)
            prototype.append(-1.0 if label == 0 else 1.0)
            classifier.append(-2.0 if label == 0 else 2.0)
            labels.append(label)
    arrays = {
        "dataset": np.asarray(datasets),
        "subject_id": np.asarray(subjects),
        "record_id": np.asarray(records),
        "window_start": np.asarray(starts),
        "analysis_scope": np.asarray(scopes),
        "prototype_score": np.asarray(prototype),
        "classifier_logit_difference": np.asarray(classifier),
    }
    if leak_labels:
        arrays["labels"] = np.asarray(labels)
    score_path = output_dir / "scores.npz"
    np.savez_compressed(score_path, **arrays)
    comparison_path = output_dir / "direction_comparison.json"
    comparison_path.write_text(
        json.dumps({"cosine": 1.0, "angle_degrees": 0.0}),
        encoding="utf-8",
    )
    correlations = {
        scope: {"support": 4, "pearson": 1.0, "spearman": 1.0}
        for scope in rows_by_scope
    }
    correlation_path = output_dir / "score_correlation.json"
    correlation_path.write_text(
        json.dumps(
            {
                "correlations": correlations,
                "equivalence": {"conclusion": "highly_equivalent"},
            }
        ),
        encoding="utf-8",
    )
    artifact = {
        "frozen": True,
        "target_labels_accessed": False,
        "diagnostic_max_batches_per_split": None,
        "source_index_sha256": sha256_file(source_index),
        "target_index_sha256": sha256_file(target_index),
        "score_sha256": sha256_file(score_path),
    }
    artifact_path = output_dir / "score_artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest = {
        "git": {"commit": "fixture", "dirty": False},
        "target_labels_accessed": False,
        "score_artifact_sha256": sha256_file(artifact_path),
        "direction_comparison_sha256": sha256_file(comparison_path),
        "score_correlation_sha256": sha256_file(correlation_path),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = {
        "role": "post_hoc_analysis",
        "protocol": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "sources": {
            "source_fixture": {
                "source_config": str(source_config_path),
                "checkpoint": str(checkpoint_path),
                "direction": str(direction_path),
                "target_dataset": "target_fixture",
                "target_index": str(target_index),
                "target_data_root": str(tmp_path),
                "output_dir": str(output_dir),
            }
        },
    }
    return config, output_dir


def test_finalize_joins_target_labels_only_after_hash_verification(tmp_path: Path) -> None:
    config, output_dir = _write_finalize_fixture(tmp_path)

    result = finalize_head_direction_analysis(
        config, source_name="source_fixture"
    )

    assert result["adaptation_time_target_labels_accessed"] is False
    assert result["post_freeze_target_evaluation_labels_accessed"] is True
    assert result["target_label_usage"] == "post-hoc analysis only"
    assert result["ranking_metrics"]["target_evaluation"][
        "prototype_score"
    ]["auroc"] == 1.0
    assert (output_dir / "split_metrics.csv").is_file()
    assert (output_dir / "score_scatter.png").is_file()


def test_finalize_rejects_target_score_label_leakage(tmp_path: Path) -> None:
    config, _ = _write_finalize_fixture(tmp_path, leak_labels=True)

    with pytest.raises(ValueError, match="leaked labels"):
        finalize_head_direction_analysis(config, source_name="source_fixture")
