#!/usr/bin/env python3
"""Train one formal or diagnostic Main M2 lambda/fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.axis_alignment import train_axis_fold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--lambda-axis", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--diagnostic-windows-per-class", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = train_axis_fold(
        config,
        fold=args.fold,
        lambda_axis=args.lambda_axis,
        device_request=args.device,
        output_override=args.output_dir,
        epoch_override=args.epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        diagnostic_windows_per_class=args.diagnostic_windows_per_class,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
