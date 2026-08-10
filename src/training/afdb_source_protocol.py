"""Frozen subject-level utilities for Revision R2 AFDB source development."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.splits import assign_subjects
from src.training.reproducibility import sha256_file


FOLD_VERSION = "afdb_source_oof_5fold_v1"
FOLD_PROTOCOL_NAMESPACE = "source_oof_5fold_v1"


@dataclass(frozen=True)
class FoldAssignment:
    dataset: str
    subject_id: str
    fold_id: int
    seed: int
    version: str


def index_subjects_without_labels(index_path: Path) -> list[str]:
    """Read only the subject column used by the label-independent assignment."""

    subjects = set()
    with Path(index_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "subject_id" not in (reader.fieldnames or []):
            raise ValueError("AFDB index has no subject_id column")
        for row in reader:
            subjects.add(row["subject_id"])
    if not subjects:
        raise ValueError("AFDB index contains no subjects")
    return sorted(subjects)


def build_fold_assignments(
    subjects: list[str], *, seed: int = 42, folds: int = 5
) -> list[FoldAssignment]:
    if folds != 5:
        raise ValueError("Revision R2 freezes exactly five folds")
    ratios = {f"fold_{fold_id}": 1.0 / folds for fold_id in range(folds)}
    assigned = assign_subjects(
        subjects,
        ratios=ratios,
        seed=seed,
        dataset="afdb",
        protocol=FOLD_PROTOCOL_NAMESPACE,
    )
    rows = [
        FoldAssignment(
            dataset="afdb",
            subject_id=subject,
            fold_id=int(assigned[subject].removeprefix("fold_")),
            seed=seed,
            version=FOLD_VERSION,
        )
        for subject in sorted(assigned)
    ]
    validate_fold_assignments(rows, expected_subjects=set(subjects), folds=folds)
    return rows


def validate_fold_assignments(
    rows: list[FoldAssignment], *, expected_subjects: set[str], folds: int = 5
) -> None:
    observed = [row.subject_id for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("fold manifest contains duplicate subjects")
    if set(observed) != expected_subjects:
        raise ValueError("fold manifest does not exactly cover AFDB subjects")
    if {row.fold_id for row in rows} != set(range(folds)):
        raise ValueError("fold manifest does not contain every fold")
    if any(
        row.dataset != "afdb" or row.version != FOLD_VERSION or row.seed != 42
        for row in rows
    ):
        raise ValueError("fold manifest violates the frozen R2 identity")


def write_fold_assignments(path: Path, rows: list[FoldAssignment]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FoldAssignment.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    temporary.replace(path)


def read_fold_assignments(path: Path) -> list[FoldAssignment]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            FoldAssignment(
                dataset=row["dataset"],
                subject_id=row["subject_id"],
                fold_id=int(row["fold_id"]),
                seed=int(row["seed"]),
                version=row["version"],
            )
            for row in csv.DictReader(handle)
        ]


def audit_fold_classes(index_path: Path, rows: list[FoldAssignment]) -> dict:
    fold_for_subject = {row.subject_id: row.fold_id for row in rows}
    counts = defaultdict(Counter)
    with Path(index_path).open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            subject = item["subject_id"]
            if subject not in fold_for_subject:
                raise ValueError(f"index subject {subject} absent from fold manifest")
            label = int(item["binary_label"])
            if label not in (0, 1):
                raise ValueError("AFDB index contains a non-binary label")
            counts[fold_for_subject[subject]][label] += 1
    details = []
    all_subjects = set(fold_for_subject)
    for fold_id in range(5):
        validation_subjects = sorted(
            subject for subject, fold in fold_for_subject.items() if fold == fold_id
        )
        training_subjects = sorted(all_subjects - set(validation_subjects))
        validation = counts[fold_id]
        training = sum(
            (counts[other] for other in range(5) if other != fold_id), Counter()
        )
        if min(validation[0], validation[1], training[0], training[1]) <= 0:
            raise ValueError(f"fold {fold_id} lacks binary class coverage")
        details.append(
            {
                "fold_id": fold_id,
                "training_subjects": training_subjects,
                "validation_subjects": validation_subjects,
                "training_class_counts": {"0": training[0], "1": training[1]},
                "validation_class_counts": {"0": validation[0], "1": validation[1]},
            }
        )
    return {"valid": True, "folds": details}


def fold_subject_partitions(
    rows: list[FoldAssignment], fold_id: int
) -> tuple[set[str], set[str]]:
    if fold_id not in range(5):
        raise ValueError("fold_id must be in [0,4]")
    validation = {row.subject_id for row in rows if row.fold_id == fold_id}
    training = {row.subject_id for row in rows if row.fold_id != fold_id}
    if not training or not validation or training & validation:
        raise ValueError("invalid AFDB fold subject partition")
    if training | validation != {row.subject_id for row in rows}:
        raise ValueError("AFDB fold partition does not cover the manifest")
    return training, validation


def median_best_epoch(best_epochs: list[int]) -> int:
    if len(best_epochs) != 5 or any(epoch <= 0 for epoch in best_epochs):
        raise ValueError("final epoch requires five positive best epochs")
    return int(statistics.median(best_epochs))


def centered_prototype_margin(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return source direction, centered scores, and training midpoint."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("prototype inputs are misaligned")
    if not np.isfinite(features).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("prototype inputs require finite features and both classes")
    prototypes = np.stack([features[labels == label].mean(axis=0) for label in (0, 1)])
    difference = prototypes[1] - prototypes[0]
    norm = np.linalg.norm(difference)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("prototype direction is degenerate")
    direction = difference / norm
    raw = features @ direction
    midpoint = float((raw[labels == 0].mean() + raw[labels == 1].mean()) / 2)
    return direction.astype(np.float32), (raw - midpoint).astype(np.float64), midpoint


def config_protocol_hash(config: dict) -> str:
    protocol = Path(config["protocol"])
    actual = sha256_file(protocol)
    if config.get("protocol_sha256") != actual:
        raise ValueError("Revision R2 protocol hash mismatch")
    return actual


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
