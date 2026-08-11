#!/usr/bin/env python3
"""Continue and supervise the remaining formal Revision R2 pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _wait_for_result(path: Path, poll_seconds: int) -> dict:
    while not path.is_file():
        print(f"phase=supervisor waiting_for={path}", flush=True)
        time.sleep(poll_seconds)
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {"best_epoch", "epochs_completed", "best_validation"}
    if not required.issubset(result):
        raise ValueError(f"incomplete fold result: {path}")
    print(
        f"phase=supervisor completed={path.parent.name} "
        f"best_epoch={result['best_epoch']} "
        f"val_f1={result['best_validation']['macro_f1']:.6f}",
        flush=True,
    )
    return result


def _run(command: list[str]) -> None:
    print(f"phase=supervisor launch={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/source_afdb_r2.json"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--wait-for-fold", type=int)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.start_fold not in range(5):
        parser.error("invalid poll interval or start fold")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["output_dir"]
    python = sys.executable

    if args.wait_for_fold is not None:
        if args.wait_for_fold not in range(5):
            parser.error("wait-for-fold must be in [0,4]")
        _wait_for_result(
            output / "folds" / f"fold_{args.wait_for_fold}" / "result.json",
            args.poll_seconds,
        )

    for fold_id in range(args.start_fold, 5):
        result_path = output / "folds" / f"fold_{fold_id}" / "result.json"
        if result_path.is_file():
            _wait_for_result(result_path, args.poll_seconds)
            continue
        _run(
            [
                python,
                "-u",
                "scripts/run_afdb_oof.py",
                "--config",
                str(args.config),
                "--fold",
                str(fold_id),
                "--device",
                args.device,
            ]
        )
        _wait_for_result(result_path, args.poll_seconds)

    oof_manifest = output / "oof" / "run_manifest.json"
    if not oof_manifest.is_file():
        _run(
            [
                python,
                "-u",
                "scripts/finalize_afdb_oof.py",
                "--config",
                str(args.config),
                "--device",
                args.device,
            ]
        )
    final_rule = output / "final_epoch_rule.json"
    if not oof_manifest.is_file() or not final_rule.is_file():
        raise FileNotFoundError("OOF finalization did not produce required artifacts")

    for seed in config["full_source"]["seeds"]:
        result_path = output / "full_source" / f"seed_{seed}" / "result.json"
        if result_path.is_file():
            print(f"phase=supervisor existing_full_source_seed={seed}", flush=True)
            continue
        _run(
            [
                python,
                "-u",
                "scripts/train_afdb_full_source.py",
                "--config",
                str(args.config),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
        )
        if not result_path.is_file():
            raise FileNotFoundError(f"full-source seed {seed} produced no result")
    print("phase=supervisor revision_r2_training_complete=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
