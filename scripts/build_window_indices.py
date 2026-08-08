#!/usr/bin/env python3
"""Build complete strict 10-second window CSV indices; do not read signals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DATASET_NAMES, create_adapter
from src.data.rhythm_mapping import load_rhythm_mapping
from src.data.splits import read_subject_splits
from src.data.window_index import index_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/windowing.json")
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_NAMES),
        help="comma-separated canonical dataset identifiers",
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        window_config = json.load(handle)
    mapping = load_rhythm_mapping()
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = set(requested) - set(DATASET_NAMES)
    if unknown:
        parser.error(f"unknown datasets: {sorted(unknown)}")

    results = []
    for dataset in requested:
        results.append(
            index_dataset(
                adapter=create_adapter(dataset, data_root=args.data_root),
                subject_splits=read_subject_splits(
                    args.index_dir / f"{dataset}_subject_splits.csv"
                ),
                output_path=args.index_dir / f"{dataset}_windows.csv",
                mapping=mapping,
                window_config=window_config,
            )
        )
    payload = {
        "window_version": window_config["version"],
        "mapping_version": mapping.version,
        "datasets": results,
    }
    with (args.index_dir / "window_index_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
