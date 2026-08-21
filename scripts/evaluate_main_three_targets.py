#!/usr/bin/env python3
"""Score, evaluate, and audit the frozen Main M2 model on three targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.main_three_targets import (  # noqa: E402
    completion_audit,
    evaluate_targets,
    score_target,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score-target", "evaluate", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "score-target":
        if not args.target:
            parser.error("score-target requires --target")
        result = score_target(
            config,
            target=args.target,
            device_request=args.device,
            output_override=args.output_dir,
            max_batches=args.max_batches,
        )
    elif args.command == "evaluate":
        result = evaluate_targets(config)
    else:
        result = completion_audit(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
