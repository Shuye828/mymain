#!/usr/bin/env python3
"""Build the independent LTAFDB-clean1h-v1 window index."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import create_adapter
from src.data.rhythm_mapping import load_rhythm_mapping
from src.data.splits import read_subject_splits
from src.data.window_index import index_dataset
from src.training.reproducibility import git_identity, sha256_file


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_clean_index(
    config_path: Path,
    *,
    output_override: Path | None = None,
    artifact_dir_override: Path | None = None,
) -> dict:
    config_path = Path(config_path)
    config = _load_json(config_path)
    if config.get("dataset") != "ltafdb":
        raise ValueError("R1 clean-1h builder only accepts dataset='ltafdb'")
    if config.get("dataset_version") != "ltaf_skip_first_hour_v1":
        raise ValueError("unexpected R1 dataset version")
    skip_seconds = float(config.get("skip_first_seconds", -1))
    if skip_seconds != 3600.0:
        raise ValueError("R1 requires the frozen skip_first_seconds=3600 rule")

    base_index = Path(config["base_index"])
    output_index = Path(output_override or config["output_index"])
    if output_index.resolve() == base_index.resolve():
        raise ValueError("clean index must not overwrite the historical index")
    base_index_hash_before = sha256_file(base_index)

    base_window_config_path = Path(config["base_window_config"])
    window_config = deepcopy(_load_json(base_window_config_path))
    window_config["version"] = config["window_version"]
    mapping = load_rhythm_mapping()
    split_path = Path(config["subject_split_manifest"])
    result = index_dataset(
        adapter=create_adapter("ltafdb", data_root=Path(config["raw_data_root"])),
        subject_splits=read_subject_splits(split_path),
        output_path=output_index,
        mapping=mapping,
        window_config=window_config,
        minimum_start_seconds=skip_seconds,
    )
    base_index_hash_after = sha256_file(base_index)
    if base_index_hash_after != base_index_hash_before:
        raise RuntimeError("historical LTAFDB index changed during R1 build")

    raw_root = Path(config["raw_data_root"]) / "ltafdb"
    protocol_path = Path("EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md")
    identity_inputs = {
        "dataset_version": config["dataset_version"],
        "base_dataset_version": config["base_dataset_version"],
        "skip_first_seconds": skip_seconds,
        "window_version": config["window_version"],
        "base_checksum_manifest_sha256": sha256_file(raw_root / "SHA256SUMS.txt"),
        "records_manifest_sha256": sha256_file(raw_root / "RECORDS"),
        "base_window_config_sha256": sha256_file(base_window_config_path),
        "rhythm_mapping_sha256": sha256_file(
            Path("configs/datasets/rhythm_mapping.json")
        ),
        "subject_split_sha256": sha256_file(split_path),
        "historical_index_sha256": base_index_hash_before,
        "revision_protocol_sha256": sha256_file(protocol_path),
    }
    manifest = {
        "frozen": True,
        "dataset": "ltafdb",
        "display_name": config["display_name"],
        "dataset_version": config["dataset_version"],
        "base_dataset_version": config["base_dataset_version"],
        "dataset_version_sha256": _canonical_hash(identity_inputs),
        "rule": {
            "skip_first_seconds": skip_seconds,
            "minimum_window_start_inclusive": True,
            "partial_boundary_windows": "excluded",
            "label_dependent": False,
            "raw_files_modified": False,
            "absolute_source_coordinates_preserved": True,
        },
        "identity_inputs": identity_inputs,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "historical_index_path": str(base_index),
        "historical_index_sha256_before": base_index_hash_before,
        "historical_index_sha256_after": base_index_hash_after,
        "output_index_path": str(output_index),
        "output_index_sha256": sha256_file(output_index),
        "labels_used_for_rule": False,
        "git": git_identity(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "build_summary": result,
    }
    output_dir = Path(artifact_dir_override or config["output_dir"])
    _write_json(output_dir / "dataset_version_manifest.json", manifest)
    _write_json(output_dir / "index_build_summary.json", result)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/ltaf_clean1h_v1.json"),
    )
    parser.add_argument("--output-index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.output_index is None) != (args.output_dir is None):
        parser.error(
            "diagnostic overrides require both --output-index and --output-dir"
        )
    result = build_clean_index(
        args.config,
        output_override=args.output_index,
        artifact_dir_override=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
