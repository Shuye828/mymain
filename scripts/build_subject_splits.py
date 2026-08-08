#!/usr/bin/env python3
"""Build deterministic label-independent subject split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import DATASET_NAMES, create_adapter
from src.data.splits import (
    assert_no_subject_leakage,
    build_subject_splits,
    write_subject_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/index"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/splits.json")
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    summary: dict[str, object] = {
        "version": config["version"],
        "seed": config["seed"],
        "datasets": {},
    }
    for dataset in DATASET_NAMES:
        target_ratios = {
            name: float(config["target_inductive"][name])
            for name in ("adaptation", "evaluation")
        }
        rows = build_subject_splits(
            dataset=dataset,
            adapter=create_adapter(dataset, data_root=args.data_root),
            source_ratios=config["source"],
            target_ratios=target_ratios,
            seed=int(config["seed"]),
            split_version=config["version"],
        )
        assert_no_subject_leakage(rows)
        write_subject_splits(
            args.output_dir / f"{dataset}_subject_splits.csv", rows
        )
        summary["datasets"][dataset] = {
            "subjects": len(rows),
            "source": dict(Counter(row.source_split for row in rows)),
            "target_inductive": dict(Counter(row.target_split for row in rows)),
            "eligible_records": sum(row.eligible_record_count for row in rows),
        }

    summary_path = args.output_dir / "subject_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
