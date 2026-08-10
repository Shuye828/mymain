#!/usr/bin/env python3
"""Select, evaluate, and summarize Stage 5C source thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.source_threshold_baseline import (
    evaluate_source_thresholds,
    select_source_thresholds,
    summarize_source_thresholds,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--source", required=True)
        child.add_argument("--output-dir", type=Path)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if args.command == "select":
        result = select_source_thresholds(
            config, source_name=args.source, output_override=args.output_dir
        )
    elif args.command == "evaluate":
        result = evaluate_source_thresholds(
            config, source_name=args.source, output_override=args.output_dir
        )
    else:
        result = summarize_source_thresholds(config, output_override=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
