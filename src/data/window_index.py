"""Strict grid-aligned source-rate window index construction."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .rhythm_mapping import RhythmMapping
from .schema import RhythmInterval
from .splits import SubjectSplit


WINDOW_INDEX_FIELDS = [
    "dataset",
    "record_id",
    "subject_id",
    "source_path",
    "fs_original",
    "channel_names",
    "start_sample",
    "end_sample",
    "window_start_seconds",
    "rhythm_label",
    "binary_label",
    "is_transition",
    "split",
    "source_split",
    "target_split",
    "target_transductive_split",
    "annotation_source",
    "mapping_version",
    "split_version",
    "window_version",
    "cpsc_boundary_version",
]


@dataclass(frozen=True)
class WindowDecision:
    start_sample: int
    end_sample: int
    interval: RhythmInterval | None
    reason: str

    @property
    def accepted(self) -> bool:
        return self.reason == "accepted"


def classify_grid_windows(
    *,
    signal_length: int,
    intervals: Sequence[RhythmInterval],
    window_samples: int,
    stride_samples: int,
    minimum_start_sample: int = 0,
) -> Iterator[WindowDecision]:
    """Classify non-overlapping/global-grid windows by full containment."""

    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride samples must be positive")
    if minimum_start_sample < 0:
        raise ValueError("minimum start sample must be non-negative")
    interval_index = 0
    for start in range(0, max(0, signal_length - window_samples + 1), stride_samples):
        end = start + window_samples
        if start < minimum_start_sample:
            yield WindowDecision(start, end, None, "before_minimum_start")
            continue
        while (
            interval_index < len(intervals)
            and intervals[interval_index].end_sample <= start
        ):
            interval_index += 1
        if interval_index >= len(intervals):
            yield WindowDecision(start, end, None, "missing_annotation")
            continue
        interval = intervals[interval_index]
        if interval.start_sample <= start and end <= interval.end_sample:
            if interval.action in {"af", "nonaf"}:
                reason = "accepted"
            elif interval.raw_token == "__UNANNOTATED__":
                reason = "unannotated"
            else:
                reason = "excluded_rhythm"
            yield WindowDecision(start, end, interval, reason)
        else:
            yield WindowDecision(start, end, None, "transition")


def index_dataset(
    *,
    adapter,
    subject_splits: dict[str, SubjectSplit],
    output_path: Path,
    mapping: RhythmMapping,
    window_config: dict,
    minimum_start_seconds: float = 0.0,
) -> dict:
    """Stream one dataset's accepted windows to CSV with exclusion statistics."""

    duration_seconds = float(window_config["duration_seconds"])
    stride_seconds = float(window_config["stride_seconds"])
    minimum_start_seconds = float(minimum_start_seconds)
    if not math.isfinite(minimum_start_seconds) or minimum_start_seconds < 0:
        raise ValueError("minimum start seconds must be finite and non-negative")
    stats: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_split_counts: Counter[str] = Counter()
    target_split_counts: Counter[str] = Counter()
    accepted_subjects: set[str] = set()
    accepted_records: set[str] = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_INDEX_FIELDS)
        writer.writeheader()
        for record_id in adapter.list_records():
            metadata = adapter.read_metadata(record_id)
            if not metadata.has_signal:
                stats["record_no_signal"] += 1
                continue
            split = subject_splits.get(metadata.subject_id)
            if split is None:
                stats["record_ineligible_subject"] += 1
                continue
            intervals = adapter.read_rhythm_intervals(record_id)
            if not intervals:
                stats["record_no_annotation_intervals"] += 1
                continue

            window_samples = int(round(metadata.fs * duration_seconds))
            stride_samples = int(round(metadata.fs * stride_seconds))
            minimum_start_sample = int(math.ceil(minimum_start_seconds * metadata.fs))
            if metadata.signal_length < window_samples:
                stats["record_shorter_than_window"] += 1
                continue
            stats["trailing_samples"] += (
                metadata.signal_length - window_samples
            ) % stride_samples
            for decision in classify_grid_windows(
                signal_length=metadata.signal_length,
                intervals=intervals,
                window_samples=window_samples,
                stride_samples=stride_samples,
                minimum_start_sample=minimum_start_sample,
            ):
                stats[f"window_{decision.reason}"] += 1
                if not decision.accepted or decision.interval is None:
                    continue
                interval = decision.interval
                binary_label = 1 if interval.action == "af" else 0
                writer.writerow(
                    {
                        "dataset": metadata.dataset,
                        "record_id": metadata.record_id,
                        "subject_id": metadata.subject_id,
                        "source_path": metadata.source_path,
                        "fs_original": metadata.fs,
                        "channel_names": json.dumps(
                            metadata.channel_names, ensure_ascii=False
                        ),
                        "start_sample": decision.start_sample,
                        "end_sample": decision.end_sample,
                        "window_start_seconds": decision.start_sample / metadata.fs,
                        "rhythm_label": interval.raw_token,
                        "binary_label": binary_label,
                        "is_transition": False,
                        "split": split.source_split,
                        "source_split": split.source_split,
                        "target_split": split.target_split,
                        "target_transductive_split": "transductive",
                        "annotation_source": interval.annotation_source,
                        "mapping_version": mapping.version,
                        "split_version": split.split_version,
                        "window_version": window_config["version"],
                        "cpsc_boundary_version": window_config["cpsc_boundary_version"],
                    }
                )
                class_counts[str(binary_label)] += 1
                source_split_counts[split.source_split] += 1
                target_split_counts[split.target_split] += 1
                accepted_subjects.add(metadata.subject_id)
                accepted_records.add(metadata.record_id)
    temporary_path.replace(output_path)
    return {
        "dataset": adapter.dataset,
        "minimum_start_seconds": minimum_start_seconds,
        "output_path": str(output_path),
        "accepted_windows": stats["window_accepted"],
        "accepted_records": len(accepted_records),
        "accepted_subjects": len(accepted_subjects),
        "class_counts": dict(class_counts),
        "source_split_window_counts": dict(source_split_counts),
        "target_split_window_counts": dict(target_split_counts),
        "statistics": dict(stats),
    }
