#!/usr/bin/env python3
"""Run CE-only source training or the required tiny overfit gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_source import train_source_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume model, optimizer, history, and early-stopping state",
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    result = train_source_experiment(
        config,
        tiny_overfit=args.tiny_overfit,
        device_request=args.device,
        epoch_override=args.epochs,
        output_override=args.output_dir,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        resume_path=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.tiny_overfit and not result.get("tiny_success", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
