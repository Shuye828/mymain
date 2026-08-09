#!/usr/bin/env python3
"""Prepare, extract, and finalize Stage 5B direction geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cross_dataset_direction_geometry import (
    extract_reference_geometry,
    finalize_geometry,
    prepare_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--output-dir", type=Path)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument("--reference", required=True)
    extract.add_argument("--device", default="auto")
    extract.add_argument("--output-dir", type=Path)
    extract.add_argument("--max-batches-per-dataset", type=int)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if args.command == "prepare":
        result = prepare_selection(config, output_override=args.output_dir)
    elif args.command == "extract":
        result = extract_reference_geometry(
            config,
            reference_name=args.reference,
            device_request=args.device,
            output_override=args.output_dir,
            max_batches_per_dataset=args.max_batches_per_dataset,
        )
    else:
        result = finalize_geometry(config, output_override=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
