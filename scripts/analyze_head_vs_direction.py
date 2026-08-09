#!/usr/bin/env python3
"""Extract and finalize Stage 5A head-versus-direction analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.head_direction_equivalence import (
    extract_head_direction_scores,
    finalize_head_direction_analysis,
)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="freeze scores without reading target labels"
    )
    extract_parser.add_argument("--config", type=Path, required=True)
    extract_parser.add_argument("--source", required=True)
    extract_parser.add_argument("--device", default="auto")
    extract_parser.add_argument("--output-dir", type=Path)
    extract_parser.add_argument("--max-batches-per-split", type=int)

    finalize_parser = subparsers.add_parser(
        "finalize", help="join labels after verifying frozen score hashes"
    )
    finalize_parser.add_argument("--config", type=Path, required=True)
    finalize_parser.add_argument("--source", required=True)
    finalize_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()
    config = _load_config(args.config)
    if args.command == "extract":
        result = extract_head_direction_scores(
            config,
            source_name=args.source,
            device_request=args.device,
            output_override=args.output_dir,
            max_batches_per_split=args.max_batches_per_split,
        )
    else:
        result = finalize_head_direction_analysis(
            config,
            source_name=args.source,
            output_override=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
