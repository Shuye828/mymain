"""Deterministic subject-level source and target protocol splits."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SubjectSplit:
    dataset: str
    subject_id: str
    source_split: str
    target_split: str
    eligible_record_count: int
    seed: int
    split_version: str


def _namespace_seed(seed: int, dataset: str, protocol: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{dataset}:{protocol}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def largest_remainder_counts(
    total: int, ratios: Mapping[str, float]
) -> dict[str, int]:
    """Allocate all subjects while staying closest to requested ratios."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative and non-empty")
    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"split ratios must sum to one, got {ratio_sum}")
    exact = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(
        ratios,
        key=lambda name: (-(exact[name] - counts[name]), name),
    )
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def assign_subjects(
    subjects: Iterable[str],
    *,
    ratios: Mapping[str, float],
    seed: int,
    dataset: str,
    protocol: str,
) -> dict[str, str]:
    """Randomly assign unique subjects without using rhythm/class labels."""

    unique = sorted(set(str(subject) for subject in subjects))
    rng = random.Random(_namespace_seed(seed, dataset, protocol))
    rng.shuffle(unique)
    counts = largest_remainder_counts(len(unique), ratios)
    assignments: dict[str, str] = {}
    offset = 0
    for split, count in counts.items():
        for subject in unique[offset : offset + count]:
            assignments[subject] = split
        offset += count
    if len(assignments) != len(unique):
        raise AssertionError("not every subject was assigned exactly once")
    return assignments


def eligible_subject_record_counts(adapter) -> dict[str, int]:
    """Count labelled signal records without inspecting binary class values."""

    counts: Counter[str] = Counter()
    for record_id in adapter.list_records():
        metadata = adapter.read_metadata(record_id)
        if not metadata.has_signal or not metadata.has_annotation:
            continue
        intervals = adapter.read_rhythm_intervals(record_id)
        if any(interval.action in {"af", "nonaf"} for interval in intervals):
            counts[metadata.subject_id] += 1
    return dict(counts)


def build_subject_splits(
    *,
    dataset: str,
    adapter,
    source_ratios: Mapping[str, float],
    target_ratios: Mapping[str, float],
    seed: int,
    split_version: str,
) -> list[SubjectSplit]:
    record_counts = eligible_subject_record_counts(adapter)
    subjects = sorted(record_counts)
    source = assign_subjects(
        subjects,
        ratios=source_ratios,
        seed=seed,
        dataset=dataset,
        protocol="source",
    )
    target = assign_subjects(
        subjects,
        ratios=target_ratios,
        seed=seed,
        dataset=dataset,
        protocol="target_inductive",
    )
    return [
        SubjectSplit(
            dataset=dataset,
            subject_id=subject,
            source_split=source[subject],
            target_split=target[subject],
            eligible_record_count=record_counts[subject],
            seed=seed,
            split_version=split_version,
        )
        for subject in subjects
    ]


def assert_no_subject_leakage(rows: Sequence[SubjectSplit]) -> None:
    """Assert one source and target assignment per dataset/subject."""

    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        key = (row.dataset, row.subject_id)
        value = (row.source_split, row.target_split)
        if key in seen and seen[key] != value:
            raise ValueError(f"conflicting subject split for {key}")
        seen[key] = value


def write_subject_splits(path: Path, rows: Sequence[SubjectSplit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "subject_id",
                "source_split",
                "target_split",
                "eligible_record_count",
                "seed",
                "split_version",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_subject_splits(path: Path) -> dict[str, SubjectSplit]:
    rows: dict[str, SubjectSplit] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            row = SubjectSplit(
                dataset=item["dataset"],
                subject_id=item["subject_id"],
                source_split=item["source_split"],
                target_split=item["target_split"],
                eligible_record_count=int(item["eligible_record_count"]),
                seed=int(item["seed"]),
                split_version=item["split_version"],
            )
            if row.subject_id in rows:
                raise ValueError(f"duplicate subject {row.subject_id!r} in {path}")
            rows[row.subject_id] = row
    return rows
