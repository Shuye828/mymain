#!/usr/bin/env python3
"""Audit the R1 clean index against the frozen historical LTAFDB index."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import create_adapter
from src.data.splits import read_subject_splits
from src.data.window_index import WINDOW_INDEX_FIELDS
from src.training.reproducibility import sha256_file


UNCHANGED_FIELDS = tuple(
    field for field in WINDOW_INDEX_FIELDS if field != "window_version"
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_counts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "historical_windows",
        "removed_first_hour_windows",
        "clean_windows",
        "clean_nonaf_windows",
        "clean_af_windows",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _filtered_rows(path: Path, minimum_seconds: float):
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["window_start_seconds"]) >= minimum_seconds:
                yield row


def audit_clean_index(config_path: Path) -> dict:
    config = _load_json(Path(config_path))
    historical_path = Path(config["base_index"])
    clean_path = Path(config["output_index"])
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "dataset_version_manifest.json"
    manifest = _load_json(manifest_path)
    minimum_seconds = float(config["skip_first_seconds"])

    errors: list[str] = []
    if sha256_file(historical_path) != manifest["historical_index_sha256_before"]:
        errors.append("historical index hash changed after clean build")
    if sha256_file(clean_path) != manifest["output_index_sha256"]:
        errors.append("clean index hash does not match version manifest")

    historical_total = 0
    removed_total = 0
    historical_record_counts = Counter()
    historical_subject_counts = Counter()
    removed_record_counts = Counter()
    removed_subject_counts = Counter()
    with historical_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            historical_total += 1
            historical_record_counts[row["record_id"]] += 1
            historical_subject_counts[row["subject_id"]] += 1
            if float(row["window_start_seconds"]) < minimum_seconds:
                removed_total += 1
                removed_record_counts[row["record_id"]] += 1
                removed_subject_counts[row["subject_id"]] += 1

    clean_total = 0
    clean_class = Counter()
    clean_record_counts = defaultdict(Counter)
    clean_subject_counts = defaultdict(Counter)
    subject_source_splits = defaultdict(set)
    subject_target_splits = defaultdict(set)
    minimum_observed = float("inf")
    sentinel = object()
    with clean_path.open(encoding="utf-8", newline="") as handle:
        clean_reader = csv.DictReader(handle)
        for expected, observed in zip_longest(
            _filtered_rows(historical_path, minimum_seconds),
            clean_reader,
            fillvalue=sentinel,
        ):
            if expected is sentinel or observed is sentinel:
                errors.append("historical-filtered and clean row counts differ")
                break
            clean_total += 1
            if any(expected[field] != observed[field] for field in UNCHANGED_FIELDS):
                errors.append(f"row identity mismatch at clean row {clean_total}")
                if len(errors) >= 100:
                    break
            start_seconds = float(observed["window_start_seconds"])
            minimum_observed = min(minimum_observed, start_seconds)
            if start_seconds < minimum_seconds:
                errors.append(f"clean row {clean_total} starts before cutoff")
            if observed["window_version"] != config["window_version"]:
                errors.append(f"clean row {clean_total} has wrong window version")
            label = observed["binary_label"]
            clean_class[label] += 1
            clean_record_counts[observed["record_id"]]["total"] += 1
            clean_record_counts[observed["record_id"]][label] += 1
            clean_subject_counts[observed["subject_id"]]["total"] += 1
            clean_subject_counts[observed["subject_id"]][label] += 1
            subject_source_splits[observed["subject_id"]].add(observed["source_split"])
            subject_target_splits[observed["subject_id"]].add(observed["target_split"])

    split_manifest = read_subject_splits(Path(config["subject_split_manifest"]))
    for subject in clean_subject_counts:
        if subject not in split_manifest:
            errors.append(f"clean subject {subject} is absent from split manifest")
        if len(subject_source_splits[subject]) != 1:
            errors.append(f"clean subject {subject} crosses source splits")
        if len(subject_target_splits[subject]) != 1:
            errors.append(f"clean subject {subject} crosses target splits")

    adapter = create_adapter("ltafdb", data_root=Path(config["raw_data_root"]))
    remaining_duration_seconds = 0.0
    for record_id in adapter.list_records():
        metadata = adapter.read_metadata(record_id)
        remaining_duration_seconds += max(
            0.0, metadata.signal_length / metadata.fs - minimum_seconds
        )

    def count_rows(keys, historical, removed, clean):
        return [
            {
                "id": key,
                "historical_windows": historical[key],
                "removed_first_hour_windows": removed[key],
                "clean_windows": clean[key]["total"],
                "clean_nonaf_windows": clean[key]["0"],
                "clean_af_windows": clean[key]["1"],
            }
            for key in sorted(keys)
        ]

    _write_counts(
        output_dir / "record_window_counts.csv",
        count_rows(
            clean_record_counts.keys(),
            historical_record_counts,
            removed_record_counts,
            clean_record_counts,
        ),
    )
    _write_counts(
        output_dir / "subject_window_counts.csv",
        count_rows(
            clean_subject_counts.keys(),
            historical_subject_counts,
            removed_subject_counts,
            clean_subject_counts,
        ),
    )
    comparison = {
        "historical_version": config["base_dataset_version"],
        "clean_version": config["dataset_version"],
        "historical_windows": historical_total,
        "removed_first_hour_accepted_windows": removed_total,
        "clean_windows": clean_total,
        "clean_class_counts": dict(clean_class),
        "clean_subjects": len(clean_subject_counts),
        "clean_records": len(clean_record_counts),
        "minimum_clean_window_start_seconds": minimum_observed,
        "remaining_duration_seconds": remaining_duration_seconds,
        "historical_index_sha256": sha256_file(historical_path),
        "clean_index_sha256": sha256_file(clean_path),
    }
    validation = {
        **comparison,
        "historical_filtered_identity_equals_clean": not errors,
        "patient_leakage_detected": any("crosses" in error for error in errors),
        "error_count": len(errors),
        "errors": errors[:100],
        "valid": not errors,
    }
    _write_json(output_dir / "old_vs_clean_summary.json", comparison)
    _write_json(output_dir / "index_validation.json", validation)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/ltaf_clean1h_v1.json"),
    )
    args = parser.parse_args()
    result = audit_clean_index(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
