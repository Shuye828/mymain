import csv
from pathlib import Path

import numpy as np
import pytest

from src.analysis.cross_dataset_direction_geometry import (
    GeometryAccumulator,
    centroid_distance_matrix,
    direction_cosine_matrix,
    prepare_selection,
    select_analysis_rows,
    validate_protocol,
)
from src.training.reproducibility import sha256_file


FIELDS = [
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


def _fixture_config(tmp_path: Path) -> dict:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("stage 5b protocol\n", encoding="utf-8")
    datasets = {}
    for dataset in ("a", "b", "c", "d"):
        index = tmp_path / f"{dataset}.csv"
        with index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for label in (0, 1):
                for sample in range(4):
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "record_id": f"r{label}",
                            "subject_id": "s1",
                            "start_sample": sample * 2000 + label * 10000,
                            "end_sample": sample * 2000 + label * 10000 + 2000,
                            "fs_original": 200,
                            "binary_label": label,
                            "rhythm_label": "fixture",
                            "source_split": "train",
                            "target_split": "evaluation",
                        }
                    )
        datasets[dataset] = {"index_path": str(index), "data_root": str(tmp_path)}
    return {
        "role": "post_hoc_analysis",
        "target_label_usage": "post-hoc mechanism analysis only",
        "protocol": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "seed": 42,
        "dataset_order": ["a", "b", "c", "d"],
        "datasets": datasets,
        "selection": {"max_windows_per_subject_per_class": 2},
        "output_dir": str(tmp_path / "output"),
    }


def test_protocol_requires_explicit_post_hoc_target_label_role(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config["target_label_usage"] = "training"
    with pytest.raises(ValueError, match="explicitly post-hoc"):
        validate_protocol(config)


def test_common_selection_is_deterministic_and_capped(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    first = select_analysis_rows(config)
    second = select_analysis_rows(config)
    assert first == second
    assert len(first) == 16
    assert all(
        sum(row.dataset == name and row.binary_label == label for row in first) == 2
        for name in config["dataset_order"]
        for label in (0, 1)
    )


def test_prepare_selection_freezes_labelled_manifest_and_hash(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    result = prepare_selection(config)
    manifest = Path(config["output_dir"]) / "selected_window_manifest.csv"
    assert result["target_labels_accessed"] is True
    assert result["target_label_usage"] == "post-hoc mechanism analysis only"
    assert result["selected_window_manifest_sha256"] == sha256_file(manifest)
    with pytest.raises(FileExistsError, match="already been prepared"):
        prepare_selection(config)


def test_geometry_accumulator_matches_window_and_subject_estimands() -> None:
    accumulator = GeometryAccumulator(2)
    accumulator.update(
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        np.array([0, 0, 1, 1]),
        ["a", "a", "a", "b"],
    )
    result = accumulator.finalize()
    assert result["class_window_counts"] == [2, 2]
    assert np.allclose(result["window_weighted"]["centroid"], [0.5, 0.5])
    assert np.allclose(
        result["window_weighted"]["direction"], [-1 / np.sqrt(2), 1 / np.sqrt(2)]
    )
    assert np.allclose(result["subject_equal"]["centroid"], [1 / 3, 2 / 3])


def test_geometry_rejects_raw_non_normalized_features() -> None:
    accumulator = GeometryAccumulator(2)
    with pytest.raises(ValueError, match="L2-normalized"):
        accumulator.update(np.array([[2.0, 0.0]]), np.array([0]), ["s"])


def test_pairwise_geometry_matrices_are_symmetric_with_expected_diagonal() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    cosine = direction_cosine_matrix(vectors)
    distance = centroid_distance_matrix(vectors)
    assert np.allclose(cosine, cosine.T)
    assert np.allclose(np.diag(cosine), 1.0)
    assert np.isclose(cosine[0, 2], -1.0)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.isclose(distance[0, 2], 2.0)
