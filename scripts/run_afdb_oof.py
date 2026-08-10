#!/usr/bin/env python3
"""Train one Revision R2 AFDB subject-level OOF fold."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.afdb_source_protocol import (
    config_protocol_hash,
    fold_subject_partitions,
    read_fold_assignments,
    validate_fold_assignments,
)
from src.training.reproducibility import sha256_file
from src.training.train_source import train_source_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--eval-windows-per-class", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    diagnostic = (
        any(
            value is not None
            for value in (
                args.max_train_batches,
                args.max_eval_batches,
                args.eval_windows_per_class,
            )
        )
        or args.tiny_overfit
    )
    if diagnostic and args.output_dir is None:
        parser.error("R2 diagnostics require an explicit --output-dir")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("role") != "afdb_source_oof" or config.get("dataset") != "afdb":
        raise ValueError("R2 OOF runner requires the frozen AFDB source config")
    config_protocol_hash(config)
    fold_path = Path(config["fold_manifest"])
    assignments = read_fold_assignments(fold_path)
    subjects = {row.subject_id for row in assignments}
    validate_fold_assignments(assignments, expected_subjects=subjects)
    training, validation = fold_subject_partitions(assignments, args.fold)

    derived = deepcopy(config)
    derived["role"] = "source"
    derived["experiment"] = f"revision_r2_afdb_oof_fold_{args.fold}"
    derived["r2_fold_id"] = args.fold
    derived["fold_manifest_sha256"] = sha256_file(fold_path)
    derived["subject_partitions"] = {
        "train": sorted(training),
        "validation": sorted(validation),
    }
    derived["output_dir"] = str(
        Path(config["output_dir"]) / "folds" / f"fold_{args.fold}"
    )
    result = train_source_experiment(
        derived,
        tiny_overfit=args.tiny_overfit,
        device_request=args.device,
        epoch_override=args.epochs,
        output_override=args.output_dir,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        resume_path=args.resume,
        evaluation_windows_per_class=args.eval_windows_per_class,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.tiny_overfit and not result.get("tiny_success", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
