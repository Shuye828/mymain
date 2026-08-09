#!/usr/bin/env python3
"""Export source-train embeddings, prototypes, and AF disease direction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representation.source_export import export_source_direction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="diagnostic only; formal exports must omit this option",
    )
    args = parser.parse_args()
    if args.max_batches is not None and args.output_dir is None:
        parser.error("--max-batches requires --output-dir")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    result = export_source_direction(
        config,
        device_request=args.device,
        output_override=args.output_dir,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
