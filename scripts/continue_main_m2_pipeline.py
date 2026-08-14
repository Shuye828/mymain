#!/usr/bin/env python3
"""Run or resume the frozen Main M2 source-only pipeline in protocol order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def valid_result(path: Path, *, fold: int | None = None, value: float | None = None) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("formal") or payload.get("target_data_accessed") is not False:
        raise ValueError(f"invalid existing formal result: {path}")
    if fold is not None and payload.get("fold") != fold:
        raise ValueError(f"existing fold result mismatch: {path}")
    if value is not None and float(payload.get("lambda_axis")) != float(value):
        raise ValueError(f"existing lambda result mismatch: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stop-after", choices=("folds", "oof", "selection", "final"), default="final")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    python = sys.executable

    for value in config["lambdas"]:
        slug = f"lambda_{float(value):.2f}".replace(".", "p")
        for fold in range(5):
            directory = output / "folds" / slug / f"fold_{fold}"
            result_path = directory / "result.json"
            if valid_result(result_path, fold=fold, value=float(value)):
                print(f"SKIP completed lambda={value} fold={fold}", flush=True)
                continue
            command = [
                python,
                "-u",
                "scripts/run_main_m2_fold.py",
                "--config",
                str(config_path),
                "--fold",
                str(fold),
                "--lambda-axis",
                str(value),
                "--device",
                args.device,
            ]
            if (directory / "last.pt").exists():
                command.append("--resume")
            run(command)
    if args.stop_after == "folds":
        return 0

    for value in config["lambdas"]:
        slug = f"lambda_{float(value):.2f}".replace(".", "p")
        summary_path = output / "oof" / slug / "summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not payload.get("formal") or payload.get("target_data_accessed") is not False:
                raise ValueError(f"invalid existing OOF summary: {summary_path}")
            print(f"SKIP completed OOF lambda={value}", flush=True)
            continue
        run(
            [
                python,
                "-u",
                "scripts/finalize_main_m2_oof.py",
                "candidate",
                "--config",
                str(config_path),
                "--lambda-axis",
                str(value),
                "--device",
                args.device,
            ]
        )
    if args.stop_after == "oof":
        return 0

    selection_path = output / "oof" / "selection_artifact.json"
    if not selection_path.exists():
        run(
            [
                python,
                "-u",
                "scripts/finalize_main_m2_oof.py",
                "select",
                "--config",
                str(config_path),
            ]
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if args.stop_after == "selection" or selection["status"] == "NO_ELIGIBLE_LAMBDA":
        print(f"M2 selection status={selection['status']}; final training not started", flush=True)
        return 0

    final_dir = output / "final_model" / f"seed_{int(config['full_source']['seed'])}"
    if valid_result(final_dir / "result.json", value=float(selection["selected_lambda"])):
        print("SKIP completed Main M2 final model", flush=True)
        return 0
    command = [
        python,
        "-u",
        "scripts/train_main_m2_final.py",
        "--config",
        str(config_path),
        "--device",
        args.device,
    ]
    if (final_dir / "final.pt").exists():
        command.append("--resume")
    run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
