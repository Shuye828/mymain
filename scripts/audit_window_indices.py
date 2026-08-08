#!/usr/bin/env python3
"""Audit window CSV integrity, split consistency, and leakage constraints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DATASET_NAMES
from src.data.splits import read_subject_splits


def audit_dataset(index_path: Path, split_path: Path, duration_seconds: float) -> dict:
    splits = read_subject_splits(split_path)
    errors: list[str] = []
    windows = 0
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_class_counts: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    current_record: str | None = None
    previous_start = -1
    completed_records: set[str] = set()
    dataset_name = ""

    with index_path.open(encoding="utf-8", newline="") as handle:
        for line_number, item in enumerate(csv.DictReader(handle), start=2):
            windows += 1
            dataset_name = item["dataset"]
            subject = item["subject_id"]
            record = item["record_id"]
            start = int(item["start_sample"])
            end = int(item["end_sample"])
            fs = float(item["fs_original"])
            label = item["binary_label"]
            expected_length = int(round(fs * duration_seconds))
            if end - start != expected_length:
                errors.append(f"line {line_number}: wrong window length")
            if start % expected_length != 0:
                errors.append(f"line {line_number}: window is off global grid")
            if item["is_transition"].lower() != "false":
                errors.append(f"line {line_number}: transition window was accepted")
            split = splits.get(subject)
            if split is None:
                errors.append(f"line {line_number}: subject missing from manifest")
            else:
                if item["source_split"] != split.source_split:
                    errors.append(f"line {line_number}: source split mismatch")
                if item["target_split"] != split.target_split:
                    errors.append(f"line {line_number}: target split mismatch")
            if record != current_record:
                if current_record is not None:
                    completed_records.add(current_record)
                if record in completed_records:
                    errors.append(
                        f"line {line_number}: record appears in multiple blocks"
                    )
                current_record = record
                previous_start = -1
            if start <= previous_start:
                errors.append(
                    f"line {line_number}: duplicate or unsorted window start"
                )
            previous_start = start
            class_counts[label] += 1
            source_counts[item["source_split"]] += 1
            source_class_counts[(item["source_split"], label)] += 1
            target_counts[item["target_split"]] += 1
            subject_counts[subject] += 1
            record_counts[record] += 1

    expected_source = {"train", "validation", "test"}
    for split_name in expected_source:
        for label in ("0", "1"):
            if source_class_counts[(split_name, label)] == 0:
                errors.append(f"source split {split_name} has no class {label}")

    return {
        "dataset": dataset_name,
        "windows": windows,
        "subjects_with_windows": len(subject_counts),
        "records_with_windows": len(record_counts),
        "class_counts": dict(class_counts),
        "source_split_counts": dict(source_counts),
        "source_split_class_counts": {
            f"{split}:{label}": count
            for (split, label), count in sorted(source_class_counts.items())
        },
        "target_split_counts": dict(target_counts),
        "min_windows_per_subject": min(subject_counts.values(), default=0),
        "max_windows_per_subject": max(subject_counts.values(), default=0),
        "error_count": len(errors),
        "errors": errors[:100],
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/windowing.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_audit/window_index_validation.json"),
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)

    results = [
        audit_dataset(
            args.index_dir / f"{dataset}_windows.csv",
            args.index_dir / f"{dataset}_subject_splits.csv",
            float(config["duration_seconds"]),
        )
        for dataset in DATASET_NAMES
    ]
    payload = {
        "all_valid": all(item["valid"] for item in results),
        "total_windows": sum(item["windows"] for item in results),
        "datasets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
