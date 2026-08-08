#!/usr/bin/env python3
"""Read and preprocess deterministic real windows from every dataset/class."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg_dataset import ECGWindowDataset, WindowRow
from src.data.registry import DATASET_NAMES


def select_rows(path: Path, per_class: int) -> list[WindowRow]:
    selected: dict[int, list[WindowRow]] = {0: [], 1: []}
    with path.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            if item["source_split"] != "train":
                continue
            label = int(item["binary_label"])
            if len(selected[label]) >= per_class:
                continue
            selected[label].append(
                WindowRow(
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
            )
            if all(len(rows) >= per_class for rows in selected.values()):
                break
    return selected[0] + selected[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/data_audit/window_pipeline_validation.json"),
    )
    args = parser.parse_args()

    results = []
    errors: list[dict[str, str]] = []
    for dataset in DATASET_NAMES:
        rows = select_rows(
            args.index_dir / f"{dataset}_windows.csv", args.per_class
        )
        source = ECGWindowDataset(rows, data_root=args.data_root, expose_label=True)
        for index, row in enumerate(rows):
            try:
                item = source[index]
                signal = item["x"].numpy()
                if signal.shape != (2, 2000):
                    raise ValueError(f"unexpected shape {signal.shape}")
                if signal.dtype != np.float32:
                    raise ValueError(f"unexpected dtype {signal.dtype}")
                if not np.isfinite(signal).all():
                    raise ValueError("processed signal contains NaN/Inf")
                results.append(
                    {
                        "dataset": dataset,
                        "record_id": row.record_id,
                        "subject_id": row.subject_id,
                        "start_sample": row.start_sample,
                        "fs_original": row.fs_original,
                        "label": row.binary_label,
                        "shape": list(signal.shape),
                        "dtype": str(signal.dtype),
                        "mean": float(signal.mean()),
                        "std": float(signal.std()),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "dataset": dataset,
                        "record_id": row.record_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    payload = {
        "valid": not errors,
        "sample_count": len(results),
        "results": results,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
