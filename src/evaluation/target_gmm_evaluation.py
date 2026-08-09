"""Final label access for already-frozen target GMM artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.ecg_dataset import load_window_rows
from src.evaluation.metrics import compute_binary_metrics
from src.training.reproducibility import git_identity, sha256_file


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _labels_for_frozen_archive(
    archive: np.lib.npyio.NpzFile,
    *,
    target_index: Path,
) -> np.ndarray:
    rows = load_window_rows([target_index])
    label_by_key: dict[tuple[str, str, int], tuple[int, str]] = {}
    for row in rows:
        key = (row.subject_id, row.record_id, row.start_sample)
        if key in label_by_key:
            raise ValueError(f"duplicate target index key {key}")
        label_by_key[key] = (row.binary_label, row.target_split)
    labels: list[int] = []
    seen: set[tuple[str, str, int]] = set()
    for subject, record, start, split in zip(
        archive["subject_id"].tolist(),
        archive["record_id"].tolist(),
        archive["window_start"].tolist(),
        archive["target_split"].tolist(),
    ):
        key = (str(subject), str(record), int(start))
        if key in seen:
            raise ValueError(f"duplicate target score key {key}")
        seen.add(key)
        if key not in label_by_key:
            raise ValueError(f"target score key absent from current index: {key}")
        label, expected_split = label_by_key[key]
        if str(split) != expected_split:
            raise ValueError(f"target split mismatch for {key}")
        labels.append(label)
    if len(seen) != len(rows):
        raise ValueError(
            f"frozen archive covers {len(seen)} of {len(rows)} target windows"
        )
    return np.asarray(labels, dtype=np.int64)


def evaluate_frozen_target_gmm(
    fit_dir: Path,
    *,
    output_path: Path | None = None,
) -> dict:
    """Read target labels only after verifying a complete frozen artifact."""

    fit_dir = Path(fit_dir)
    manifest_path = fit_dir / "run_manifest.json"
    gmm_path = fit_dir / "gmm_artifact.json"
    score_path = fit_dir / "target_scores.npz"
    manifest = _load_json(manifest_path)
    artifact = _load_json(gmm_path)
    if not artifact.get("frozen") or artifact.get("labels_accessed") is not False:
        raise ValueError("target GMM artifact was not frozen label-free")
    if manifest.get("labels_accessed") is not False:
        raise ValueError("fit manifest does not prove label-free adaptation")
    if manifest.get("diagnostic_max_batches") is not None:
        raise ValueError("formal evaluation rejects diagnostic target artifacts")
    if sha256_file(score_path) != artifact.get("target_score_sha256"):
        raise ValueError("target score archive hash mismatch")
    if sha256_file(gmm_path) != manifest.get("gmm_artifact_sha256"):
        raise ValueError("target GMM artifact hash mismatch")
    config = manifest["config"]
    target_index = Path(config["target_index"])
    if sha256_file(target_index) != artifact.get("target_index_sha256"):
        raise ValueError("target index hash changed after GMM freezing")

    archive = np.load(score_path)
    forbidden = {"label", "labels", "binary_label"} & set(archive.files)
    if forbidden:
        raise ValueError(f"target score archive leaked labels: {sorted(forbidden)}")
    required = {
        "dataset",
        "subject_id",
        "record_id",
        "window_start",
        "target_split",
        "source_classifier_probability",
        "direction_score",
        "inductive_gmm_af_probability",
        "transductive_gmm_af_probability",
    }
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"target score archive missing fields: {sorted(missing)}")
    labels = _labels_for_frozen_archive(archive, target_index=target_index)
    splits = archive["target_split"]
    evaluation_mask = splits == "evaluation"
    if not evaluation_mask.any():
        raise ValueError("frozen archive contains no inductive evaluation rows")

    source_threshold = float(
        artifact["protocols"]["inductive_holdout"]["gmm"][
            "source_fixed_threshold"
        ]
    )
    classifier = archive["source_classifier_probability"]
    direction_scores = archive["direction_score"]
    inductive_gmm = archive["inductive_gmm_af_probability"]
    transductive_gmm = archive["transductive_gmm_af_probability"]
    protocols = {
        "inductive_holdout": {
            "evaluation_split": "evaluation",
            "support": int(evaluation_mask.sum()),
            "gmm_reliable": artifact["protocols"]["inductive_holdout"][
                "gmm"
            ]["reliable"],
            "gmm_reliability_failures": artifact["protocols"][
                "inductive_holdout"
            ]["gmm"]["reliability_failures"],
            "B2_source_classifier": compute_binary_metrics(
                labels[evaluation_mask], classifier[evaluation_mask]
            ),
            "B3_source_direction_fixed_threshold": compute_binary_metrics(
                labels[evaluation_mask],
                direction_scores[evaluation_mask],
                threshold=source_threshold,
            ),
            "B4_target_gmm": compute_binary_metrics(
                labels[evaluation_mask], inductive_gmm[evaluation_mask]
            ),
        },
        "transductive": {
            "evaluation_split": "transductive_all",
            "support": int(len(labels)),
            "gmm_reliable": artifact["protocols"]["transductive"]["gmm"][
                "reliable"
            ],
            "gmm_reliability_failures": artifact["protocols"]["transductive"][
                "gmm"
            ]["reliability_failures"],
            "B2_source_classifier": compute_binary_metrics(labels, classifier),
            "B3_source_direction_fixed_threshold": compute_binary_metrics(
                labels, direction_scores, threshold=source_threshold
            ),
            "B4_target_gmm": compute_binary_metrics(labels, transductive_gmm),
        },
    }
    result = {
        "source_dataset": artifact["source_dataset"],
        "target_dataset": artifact["target_dataset"],
        "adaptation_labels_accessed": False,
        "evaluation_labels_accessed_after_freeze": True,
        "fit_git": manifest["git"],
        "evaluation_git": git_identity(),
        "target_index_sha256": artifact["target_index_sha256"],
        "target_score_sha256": artifact["target_score_sha256"],
        "gmm_artifact_sha256": sha256_file(gmm_path),
        "protocols": protocols,
    }
    resolved_output = output_path or fit_dir / "evaluation_result.json"
    _write_json(resolved_output, result)
    return result
