#!/usr/bin/env python3
"""Build record and raw-rhythm inventories without creating ECG windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DATASET_NAMES, create_adapter


def build_inventory(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return record-level and aggregated rhythm-level inventory rows."""

    record_rows: list[dict[str, Any]] = []
    rhythm_totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset in DATASET_NAMES:
        adapter = create_adapter(dataset, data_root=data_root)
        for record_id in adapter.list_records():
            metadata = adapter.read_metadata(record_id)
            intervals = adapter.read_rhythm_intervals(record_id)
            record_rows.append(
                {
                    "dataset": dataset,
                    "record_id": record_id,
                    "subject_id": metadata.subject_id,
                    "source_path": metadata.source_path,
                    "fs": metadata.fs,
                    "channel_count": len(metadata.channel_names),
                    "channel_names": json.dumps(
                        metadata.channel_names, ensure_ascii=False
                    ),
                    "signal_length": metadata.signal_length,
                    "duration_seconds": metadata.duration_seconds,
                    "has_signal": metadata.has_signal,
                    "has_annotation": metadata.has_annotation,
                    "annotation_source": metadata.annotation_source or "",
                    "interval_count": len(intervals),
                }
            )
            if not intervals:
                empty_token = (
                    "__NO_SIGNAL__"
                    if not metadata.has_signal
                    else "__NO_ANNOTATION__"
                )
                key = (dataset, empty_token, "exclude")
                item = rhythm_totals.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "raw_token": empty_token,
                        "action": "exclude",
                        "record_ids": set(),
                        "subject_ids": set(),
                        "interval_count": 0,
                        "sample_count": 0,
                        "duration_seconds": 0.0,
                    },
                )
                item["record_ids"].add(record_id)
                item["subject_ids"].add(metadata.subject_id)
            for interval in intervals:
                key = (dataset, interval.raw_token, interval.action)
                item = rhythm_totals.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "raw_token": interval.raw_token,
                        "action": interval.action,
                        "record_ids": set(),
                        "subject_ids": set(),
                        "interval_count": 0,
                        "sample_count": 0,
                        "duration_seconds": 0.0,
                    },
                )
                item["record_ids"].add(record_id)
                item["subject_ids"].add(metadata.subject_id)
                item["interval_count"] += 1
                item["sample_count"] += interval.length
                item["duration_seconds"] += interval.length / metadata.fs

    rhythm_rows: list[dict[str, Any]] = []
    for item in rhythm_totals.values():
        rhythm_rows.append(
            {
                "dataset": item["dataset"],
                "raw_token": item["raw_token"],
                "action": item["action"],
                "record_count": len(item["record_ids"]),
                "subject_count": len(item["subject_ids"]),
                "interval_count": item["interval_count"],
                "sample_count": item["sample_count"],
                "duration_seconds": item["duration_seconds"],
            }
        )
    rhythm_rows.sort(key=lambda row: (row["dataset"], row["raw_token"]))
    return record_rows, rhythm_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty inventory: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/data_audit")
    )
    args = parser.parse_args()

    record_rows, rhythm_rows = build_inventory(args.data_root)
    write_csv(args.output_dir / "record_inventory.csv", record_rows)
    write_csv(args.output_dir / "rhythm_inventory.csv", rhythm_rows)
    summary = {
        "record_count": len(record_rows),
        "rhythm_row_count": len(rhythm_rows),
        "datasets": {
            dataset: sum(row["dataset"] == dataset for row in record_rows)
            for dataset in DATASET_NAMES
        },
    }
    with (args.output_dir / "inventory_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
