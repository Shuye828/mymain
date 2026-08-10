"""PyTorch dataset backed by rebuildable window CSV indices."""

from __future__ import annotations

import csv
import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .preprocessing import PreprocessingConfig, preprocess_ecg
from .registry import create_adapter


@dataclass(frozen=True)
class WindowRow:
    dataset: str
    record_id: str
    subject_id: str
    start_sample: int
    end_sample: int
    fs_original: float
    binary_label: int
    rhythm_label: str
    source_split: str
    target_split: str


def _parse_row(item: dict[str, str]) -> WindowRow:
    label = int(item["binary_label"])
    if label not in (0, 1):
        raise ValueError(f"invalid binary label {label}")
    return WindowRow(
        dataset=item["dataset"],
        record_id=item["record_id"],
        subject_id=item["subject_id"],
        start_sample=int(item["start_sample"]),
        end_sample=int(item["end_sample"]),
        fs_original=float(item["fs_original"]),
        binary_label=label,
        rhythm_label=item["rhythm_label"],
        source_split=item["source_split"],
        target_split=item["target_split"],
    )


def _row_priority(row: WindowRow, seed: int) -> int:
    payload = (
        f"{seed}:{row.dataset}:{row.subject_id}:{row.record_id}:"
        f"{row.start_sample}:{row.binary_label}"
    )
    return int.from_bytes(
        hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big"
    )


def load_window_rows(
    paths: Iterable[Path],
    *,
    source_split: str | None = None,
    target_split: str | None = None,
    include_subjects: set[str] | None = None,
    max_windows_per_subject_per_class: int | None = None,
    seed: int = 42,
) -> list[WindowRow]:
    """Stream indices and optionally retain a deterministic bounded subset."""

    if (
        max_windows_per_subject_per_class is not None
        and max_windows_per_subject_per_class <= 0
    ):
        raise ValueError("window cap must be positive")
    if target_split is not None and max_windows_per_subject_per_class is not None:
        raise ValueError(
            "target selection cannot use max_windows_per_subject_per_class "
            "because it would inspect target labels"
        )
    rows: list[WindowRow] = []
    heaps: dict[tuple[str, str, int], list[tuple[int, str, WindowRow]]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                row = _parse_row(item)
                if (
                    include_subjects is not None
                    and row.subject_id not in include_subjects
                ):
                    continue
                if source_split is not None and row.source_split != source_split:
                    continue
                if target_split is not None and row.target_split != target_split:
                    continue
                if max_windows_per_subject_per_class is None:
                    rows.append(row)
                    continue
                group = (row.dataset, row.subject_id, row.binary_label)
                heap = heaps.setdefault(group, [])
                priority = _row_priority(row, seed)
                tie = f"{row.record_id}:{row.start_sample}"
                entry = (-priority, tie, row)
                if len(heap) < max_windows_per_subject_per_class:
                    heapq.heappush(heap, entry)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, entry)
    if max_windows_per_subject_per_class is not None:
        rows = [entry[2] for heap in heaps.values() for entry in heap]
    rows.sort(
        key=lambda row: (
            row.dataset,
            row.subject_id,
            row.record_id,
            row.start_sample,
        )
    )
    return rows


def load_unlabeled_target_rows(
    paths: Iterable[Path], *, target_split: str
) -> list[WindowRow]:
    """Load target inputs without reading either target-label CSV field."""

    rows: list[WindowRow] = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                if item["target_split"] != target_split:
                    continue
                rows.append(
                    WindowRow(
                        dataset=item["dataset"],
                        record_id=item["record_id"],
                        subject_id=item["subject_id"],
                        start_sample=int(item["start_sample"]),
                        end_sample=int(item["end_sample"]),
                        fs_original=float(item["fs_original"]),
                        binary_label=-1,
                        rhythm_label="__HIDDEN_TARGET_LABEL__",
                        source_split=item["source_split"],
                        target_split=item["target_split"],
                    )
                )
    rows.sort(
        key=lambda row: (
            row.dataset,
            row.subject_id,
            row.record_id,
            row.start_sample,
        )
    )
    return rows


class ECGWindowDataset(Dataset):
    """Read source segments lazily and return `[2, 2000]` tensors."""

    def __init__(
        self,
        rows: Sequence[WindowRow],
        *,
        data_root: Path = Path("data/raw"),
        preprocessing: PreprocessingConfig = PreprocessingConfig(),
        expose_label: bool = True,
    ) -> None:
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.preprocessing = preprocessing
        self.expose_label = expose_label
        self._adapters: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _adapter(self, dataset: str):
        if dataset not in self._adapters:
            self._adapters[dataset] = create_adapter(dataset, data_root=self.data_root)
        return self._adapters[dataset]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if self.expose_label and row.binary_label not in (0, 1):
            raise ValueError("requested visible label from an unlabeled target row")
        raw = self._adapter(row.dataset).read_signal(
            row.record_id, row.start_sample, row.end_sample
        )
        signal = preprocess_ecg(
            raw,
            fs=row.fs_original,
            config=self.preprocessing,
        )
        return {
            "x": torch.from_numpy(signal),
            "y": torch.tensor(
                row.binary_label if self.expose_label else -1, dtype=torch.long
            ),
            "metadata": {
                "dataset": row.dataset,
                "subject_id": row.subject_id,
                "record_id": row.record_id,
                "window_start": row.start_sample,
                "fs_original": row.fs_original,
                "target_split": row.target_split,
            },
        }


def build_subject_class_balanced_sampler(
    rows: Sequence[WindowRow],
    *,
    seed: int = 42,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Give every observed subject/class group equal total sampling mass."""

    if not rows:
        raise ValueError("cannot sample an empty window collection")
    counts = Counter((row.dataset, row.subject_id, row.binary_label) for row in rows)
    weights = [
        1.0 / counts[(row.dataset, row.subject_id, row.binary_label)] for row in rows
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(rows) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )
