#!/usr/bin/env python3
"""Build and audit the frozen Revision R2 AFDB subject folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.afdb_source_protocol import (
    audit_fold_classes,
    build_fold_assignments,
    config_protocol_hash,
    index_subjects_without_labels,
    write_fold_assignments,
    write_json,
)
from src.training.reproducibility import git_identity, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/source_afdb_r2.json")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_protocol_hash(config)
    index_path = Path(config["index_path"])
    subjects = index_subjects_without_labels(index_path)
    rows = build_fold_assignments(subjects, seed=int(config["fold_assignment_seed"]))
    fold_path = Path(config["fold_manifest"])
    write_fold_assignments(fold_path, rows)
    audit = audit_fold_classes(index_path, rows)
    output = Path(config["output_dir"])
    manifest = {
        "frozen": True,
        "labels_used_for_assignment": False,
        "target_data_accessed": False,
        "subject_count": len(subjects),
        "index_sha256": sha256_file(index_path),
        "fold_manifest_sha256": sha256_file(fold_path),
        "protocol_sha256": sha256_file(Path(config["protocol"])),
        "git": git_identity(),
    }
    write_json(output / "fold_manifest.json", manifest)
    write_json(output / "fold_audit.json", {**audit, **manifest})
    print(json.dumps({**manifest, **audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
