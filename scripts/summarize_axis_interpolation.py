#!/usr/bin/env python3
"""Audit and summarize frozen Main M1 outputs without reading raw signals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.axis_interpolation import FORBIDDEN, dose_response_means
from src.training.reproducibility import git_identity, sha256_file


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_outputs(config: dict, root: Path) -> dict:
    oof_dir = root / "oof"
    oof_manifest = load_json(oof_dir / "run_manifest.json")
    oof_path = oof_dir / "normalized_oof_features_and_scores.npz"
    oof = np.load(oof_path)
    oof_ids = list(
        zip(
            map(str, oof["dataset"]),
            map(str, oof["subject_id"]),
            map(str, oof["record_id"]),
            map(int, oof["window_start"]),
        )
    )
    feature_norms = np.linalg.norm(oof["features"], axis=1)
    selection_path = oof_dir / "selection_artifact.json"
    selection = load_json(selection_path)
    final_dir = root / "final_axis"
    final_path = final_dir / "final_directions.json"
    final = load_json(final_path)
    final_manifest = load_json(final_dir / "run_manifest.json")

    target_audits = {}
    for target, entry in config["targets"].items():
        target_dir = root / "targets" / target
        artifact_path = target_dir / "score_artifact.json"
        artifact = load_json(artifact_path)
        extraction = load_json(target_dir / "extraction_manifest.json")
        score_path = target_dir / "scores.npz"
        archive = np.load(score_path)
        identities = list(
            zip(
                map(str, archive["dataset"]),
                map(str, archive["subject_id"]),
                map(str, archive["record_id"]),
                map(int, archive["window_start"]),
            )
        )
        score_fields = [name for name in archive.files if name.startswith("score_")]
        target_audits[target] = {
            "expected_windows": int(entry["expected_evaluation_windows"]),
            "actual_windows": len(identities),
            "unique_windows": len(set(identities)),
            "formal": artifact.get("formal") is True,
            "target_labels_accessed_during_scoring": artifact.get(
                "target_labels_accessed"
            ),
            "forbidden_archive_fields": sorted(FORBIDDEN & set(archive.files)),
            "score_field_count": len(score_fields),
            "scores_finite": bool(
                all(np.isfinite(archive[name]).all() for name in score_fields)
            ),
            "score_sha256": artifact["score_sha256"],
            "score_hash_valid": sha256_file(score_path) == artifact["score_sha256"],
            "artifact_hash_valid": sha256_file(artifact_path)
            == extraction["score_artifact_sha256"],
            "extraction_git": extraction["git"],
        }

    gates = {
        "oof_formal": oof_manifest.get("formal") is True,
        "oof_windows_exact": len(oof_ids) == config["source"]["expected_windows"],
        "oof_identities_unique": len(oof_ids) == len(set(oof_ids)),
        "oof_archive_hash_valid": sha256_file(oof_path)
        == oof_manifest["archive_sha256"],
        "oof_features_finite": bool(np.isfinite(oof["features"]).all()),
        "selection_formal_source_only": selection.get("formal") is True
        and selection.get("target_data_accessed") is False,
        "final_axis_formal_source_only": final.get("formal") is True
        and final.get("target_data_accessed") is False,
        "final_axis_hash_valid": sha256_file(final_path)
        == final_manifest["artifact_sha256"],
        "all_target_gates_pass": all(
            audit["expected_windows"]
            == audit["actual_windows"]
            == audit["unique_windows"]
            and audit["formal"]
            and audit["target_labels_accessed_during_scoring"] is False
            and not audit["forbidden_archive_fields"]
            and audit["score_field_count"] == len(config["alphas"])
            and audit["scores_finite"]
            and audit["score_hash_valid"]
            and audit["artifact_hash_valid"]
            for audit in target_audits.values()
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"Main M1 completion audit failed: {gates}")
    return {
        "formal": True,
        "all_gates_pass": True,
        "gates": gates,
        "oof": {
            "windows": len(oof_ids),
            "feature_shape": list(oof["features"].shape),
            "feature_norm_min": float(feature_norms.min()),
            "feature_norm_max": float(feature_norms.max()),
            "archive_sha256": sha256_file(oof_path),
            "git": oof_manifest["git"],
        },
        "selection": {
            "alpha": selection["selected_alpha"],
            "threshold": selection["selected_threshold"],
            "artifact_sha256": sha256_file(selection_path),
        },
        "final_axis": {
            "source_windows": final["source_windows"],
            "head_prototype_cosine": final["head_prototype_cosine"],
            "artifact_sha256": sha256_file(final_path),
        },
        "targets": target_audits,
        "total_target_windows": sum(
            item["actual_windows"] for item in target_audits.values()
        ),
    }


def make_plot(
    path: Path, source_rows: list[dict], target_rows: list[dict], per_target: dict
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    alphas = [row["alpha"] for row in target_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(alphas, [row["auroc"] for row in source_rows], "o-", label="AUROC")
    axes[0, 0].plot(alphas, [row["balanced_accuracy"] for row in source_rows], "s-", label="BACC")
    axes[0, 0].set_title("AFDB OOF source-only selection")
    for target, rows in per_target.items():
        axes[0, 1].plot(alphas, [row["auroc"] for row in rows], "o-", label=target)
        axes[1, 0].plot(alphas, [row["auprc"] for row in rows], "o-", label=target)
    axes[0, 1].set_title("Target AUROC dose response")
    axes[1, 0].set_title("Target AUPRC dose response")
    for metric, marker in (("mean_balanced_accuracy", "o"), ("mean_macro_f1", "s"), ("mean_mcc", "^")):
        axes[1, 1].plot(alphas, [row[metric] for row in target_rows], marker + "-", label=metric.removeprefix("mean_"))
    axes[1, 1].set_title("Three-target mean operating metrics")
    for axis in axes.flat:
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xlabel("alpha (0=head, 1=prototype)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Main M1 — frozen disease-axis interpolation dose response")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    root = Path(config["output_dir"])
    result_path = root / "analysis_result.json"
    result = load_json(result_path)
    source = load_json(root / "oof" / "selection_artifact.json")["alphas"]
    target_rows = dose_response_means(result["per_target"])
    audit = audit_outputs(config, root)
    audit["analysis_result_sha256"] = sha256_file(result_path)
    audit["analysis_git"] = load_json(root / "run_manifest.json")["git"]
    write_csv(root / "dose_response_means.csv", target_rows)
    write_json(root / "completion_audit.json", audit)
    make_plot(root / "m1_dose_response.png", source, target_rows, result["per_target"])
    manifest = {
        "formal": True,
        "git": git_identity(),
        "analysis_result_sha256": sha256_file(result_path),
        "dose_response_means_sha256": sha256_file(root / "dose_response_means.csv"),
        "completion_audit_sha256": sha256_file(root / "completion_audit.json"),
        "figure_sha256": sha256_file(root / "m1_dose_response.png"),
    }
    write_json(root / "summary_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
