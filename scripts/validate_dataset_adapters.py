#!/usr/bin/env python3
"""Validate every source record without preprocessing or window creation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DATASET_NAMES, create_adapter
from src.data.rhythm_intervals import assert_complete_coverage, assert_interval_bounds


def validate_dataset(adapter: Any, signal_samples: int) -> dict[str, Any]:
    records = adapter.list_records()
    errors: list[dict[str, str]] = []
    subject_ids: set[str] = set()
    fs_counts: Counter[str] = Counter()
    channel_counts: Counter[str] = Counter()
    channel_name_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    signal_count = 0
    annotation_count = 0
    signal_slice_count = 0

    for record_id in records:
        try:
            metadata = adapter.read_metadata(record_id)
            if metadata.fs <= 0:
                raise ValueError("non-positive sampling frequency")
            if metadata.signal_length < 0:
                raise ValueError("negative signal length")
            subject_ids.add(metadata.subject_id)
            fs_counts[str(metadata.fs)] += 1
            channel_counts[str(len(metadata.channel_names))] += 1
            channel_name_counts[json.dumps(metadata.channel_names)] += 1
            signal_count += int(metadata.has_signal)
            annotation_count += int(metadata.has_annotation)

            intervals = adapter.read_rhythm_intervals(record_id)
            assert_interval_bounds(intervals, metadata.signal_length)
            if metadata.has_annotation and metadata.has_signal:
                assert_complete_coverage(intervals, metadata.signal_length)
            for interval in intervals:
                action_counts[interval.action] += 1
                token_counts[interval.raw_token] += 1

            if metadata.has_signal and signal_samples > 0:
                end = min(metadata.signal_length, signal_samples)
                if end <= 0:
                    raise ValueError("signal-bearing record has zero length")
                signal = adapter.read_signal(record_id, 0, end)
                expected = (len(metadata.channel_names), end)
                if signal.shape != expected:
                    raise ValueError(
                        f"signal slice shape {signal.shape}, expected {expected}"
                    )
                if not np.isfinite(signal).all():
                    raise ValueError("signal slice contains NaN/Inf")
                signal_slice_count += 1
        except Exception as exc:
            errors.append(
                {
                    "record_id": record_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "dataset": adapter.dataset,
        "record_count": len(records),
        "subject_count": len(subject_ids),
        "signal_record_count": signal_count,
        "annotated_record_count": annotation_count,
        "validated_signal_slice_count": signal_slice_count,
        "sampling_rates": dict(fs_counts),
        "channel_counts": dict(channel_counts),
        "channel_names": dict(channel_name_counts),
        "rhythm_interval_actions": dict(action_counts),
        "raw_rhythm_interval_counts": dict(token_counts),
        "error_count": len(errors),
        "errors": errors,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_audit/adapter_validation.json"))
    parser.add_argument(
        "--signal-samples",
        type=int,
        default=32,
        help="raw samples read per signal-bearing record; 0 disables signal reads",
    )
    args = parser.parse_args()

    results = [
        validate_dataset(
            create_adapter(dataset, data_root=args.data_root), args.signal_samples
        )
        for dataset in DATASET_NAMES
    ]
    payload = {
        "all_valid": all(result["valid"] for result in results),
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
