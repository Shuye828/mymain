#!/usr/bin/env python3
"""Finalize one Main M2 OOF candidate or freeze source-only lambda selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.axis_alignment_oof import finalize_lambda_oof, select_m2_lambda


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("candidate", "select"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lambda-axis", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "candidate":
        if args.lambda_axis is None:
            parser.error("candidate requires --lambda-axis")
        result = finalize_lambda_oof(
            config,
            lambda_axis=args.lambda_axis,
            device_request=args.device,
            output_override=args.output_dir,
        )
    else:
        if args.lambda_axis is not None:
            parser.error("select does not accept --lambda-axis")
        result = select_m2_lambda(config, output_override=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
