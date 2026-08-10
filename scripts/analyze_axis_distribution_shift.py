#!/usr/bin/env python3
"""Extract, finalize, and summarize Stage 5D disease-axis shift analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.axis_distribution_shift import (
    extract_axis_scores,
    finalize_axis_distribution,
    summarize_axis_distribution,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--reference", required=True)
        child.add_argument("--output-dir", type=Path)
        if command == "extract":
            child.add_argument("--device", default="auto")
            child.add_argument("--max-batches-per-dataset", type=int)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if args.command == "extract":
        result = extract_axis_scores(
            config,
            reference=args.reference,
            device_request=args.device,
            output_override=args.output_dir,
            max_batches_per_dataset=args.max_batches_per_dataset,
        )
    elif args.command == "finalize":
        result = finalize_axis_distribution(
            config, reference=args.reference, output_override=args.output_dir
        )
    else:
        result = summarize_axis_distribution(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
