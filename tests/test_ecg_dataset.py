import csv
from pathlib import Path

import numpy as np

import src.data.ecg_dataset as dataset_module
from src.data.ecg_dataset import (
    ECGWindowDataset,
    WindowRow,
    load_unlabeled_target_rows,
    load_window_rows,
)


def _write_index(path: Path) -> None:
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
        for subject in ("a", "b"):
            for label in (0, 1):
                for index in range(5):
                    writer.writerow(
                        {
                            "dataset": "cpsc2021",
                            "record_id": f"data_{subject}_{index}",
                            "subject_id": subject,
                            "start_sample": index * 2000,
                            "end_sample": (index + 1) * 2000,
                            "fs_original": 200,
                            "binary_label": label,
                            "rhythm_label": "fixture",
                            "source_split": "train",
                            "target_split": "adaptation",
                        }
                    )


def test_window_cap_is_deterministic_per_subject_and_class(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    _write_index(path)

    first = load_window_rows(
        [path], source_split="train", max_windows_per_subject_per_class=2, seed=42
    )
    second = load_window_rows(
        [path], source_split="train", max_windows_per_subject_per_class=2, seed=42
    )

    assert first == second
    assert len(first) == 8
    assert {
        (row.subject_id, row.binary_label): sum(
            candidate.subject_id == row.subject_id
            and candidate.binary_label == row.binary_label
            for candidate in first
        )
        for row in first
    } == {("a", 0): 2, ("a", 1): 2, ("b", 0): 2, ("b", 1): 2}


def test_target_loader_removes_labels_and_forbids_class_cap(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"
    _write_index(path)

    rows = load_unlabeled_target_rows([path], target_split="adaptation")

    assert rows
    assert {row.binary_label for row in rows} == {-1}
    assert {row.rhythm_label for row in rows} == {"__HIDDEN_TARGET_LABEL__"}
    try:
        load_window_rows(
            [path],
            target_split="adaptation",
            max_windows_per_subject_per_class=2,
        )
    except ValueError as exc:
        assert "would inspect target labels" in str(exc)
    else:
        raise AssertionError("target class-aware cap was unexpectedly allowed")


def test_dataset_returns_processed_shape_and_hides_target_label(monkeypatch) -> None:
    class FakeAdapter:
        def read_signal(self, record_id: str, start: int, end: int) -> np.ndarray:
            time = np.arange(end - start) / 200.0
            return np.stack(
                [np.sin(2 * np.pi * 5 * time), np.cos(2 * np.pi * 7 * time)]
            )

    monkeypatch.setattr(
        dataset_module, "create_adapter", lambda dataset, data_root: FakeAdapter()
    )
    row = WindowRow(
        dataset="cpsc2021",
        record_id="data_1_1",
        subject_id="1",
        start_sample=0,
        end_sample=2000,
        fs_original=200,
        binary_label=1,
        rhythm_label="AFp_AF",
        source_split="train",
        target_split="adaptation",
    )

    item = ECGWindowDataset([row], expose_label=False)[0]

    assert tuple(item["x"].shape) == (2, 2000)
    assert item["x"].dtype.is_floating_point
    assert item["y"].item() == -1
    assert "binary_label" not in item["metadata"]
