#!/usr/bin/env python3
"""Compare label-free LTAFDB signal quality in 0-1 h versus >=1 h."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.ltaf_signal_quality import (
    QUALITY_METRICS,
    deterministic_window_starts,
    quality_metrics_by_channel,
)
from src.data.registry import create_adapter
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


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "median": float(np.nanmedian(array)),
        "q25": float(np.nanquantile(array, 0.25)),
        "q75": float(np.nanquantile(array, 0.75)),
    }


def audit_quality(
    config_path: Path,
    *,
    output_override: Path | None = None,
    max_records: int | None = None,
    windows_override: int | None = None,
) -> dict:
    config_path = Path(config_path)
    config = _load_json(config_path)
    quality = config["quality_audit"]
    skip_seconds = float(config["skip_first_seconds"])
    window_seconds = float(quality["window_seconds"])
    sample_count = int(
        quality["windows_per_period_per_record"]
        if windows_override is None
        else windows_override
    )
    if sample_count <= 0:
        raise ValueError("quality windows per period must be positive")
    adapter = create_adapter("ltafdb", data_root=Path(config["raw_data_root"]))
    record_ids = adapter.list_records()
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        record_ids = record_ids[:max_records]

    output_dir = Path(output_override or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    record_period_channel = defaultdict(lambda: defaultdict(list))
    started = time.perf_counter()
    for record_number, record_id in enumerate(record_ids, start=1):
        metadata = adapter.read_metadata(record_id)
        window_samples = int(round(window_seconds * metadata.fs))
        cutoff_sample = int(round(skip_seconds * metadata.fs))
        if metadata.signal_length < cutoff_sample + window_samples:
            raise ValueError(f"record {record_id} has no post-cutoff quality window")
        early_candidates = range(0, cutoff_sample - window_samples + 1, window_samples)
        late_candidates = range(
            cutoff_sample,
            metadata.signal_length - window_samples + 1,
            window_samples,
        )
        early_starts = list(early_candidates)[:sample_count]
        late_starts = deterministic_window_starts(
            late_candidates,
            count=sample_count,
            seed=int(quality["seed"]),
            record_id=record_id,
        )
        signal = adapter.read_signal(record_id)
        for period, starts in (("0-1h", early_starts), (">=1h", late_starts)):
            for start in starts:
                segment = signal[:, start : start + window_samples]
                channel_metrics = quality_metrics_by_channel(
                    segment,
                    fs=metadata.fs,
                    extreme_amplitude_mv=float(quality["extreme_amplitude_mv"]),
                    extreme_first_difference_mv=float(
                        quality["extreme_first_difference_mv"]
                    ),
                    high_frequency_band_hz=tuple(quality["high_frequency_band_hz"]),
                    reference_power_band_hz=tuple(quality["reference_power_band_hz"]),
                )
                for channel_index, metrics in enumerate(channel_metrics):
                    key = (record_id, period, channel_index)
                    for metric, value in metrics.items():
                        record_period_channel[key][metric].append(value)
        print(
            f"phase=ltaf_quality record={record_number}/{len(record_ids)} "
            f"record_id={record_id} seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )

    for (record_id, period, channel_index), metric_values in sorted(
        record_period_channel.items()
    ):
        row = {
            "record_id": record_id,
            "subject_id": record_id,
            "period": period,
            "channel_index": channel_index,
            "sampled_windows": len(next(iter(metric_values.values()))),
        }
        for metric in QUALITY_METRICS:
            for statistic, value in _aggregate(metric_values[metric]).items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)

    csv_path = output_dir / "ltaf_quality_before_after.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_record = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        for metric in QUALITY_METRICS:
            per_record[row["record_id"]][row["period"]].setdefault(metric, []).append(
                float(row[f"{metric}_median"])
            )
    paired = {metric: {"0-1h": [], ">=1h": []} for metric in QUALITY_METRICS}
    for periods in per_record.values():
        for metric in QUALITY_METRICS:
            for period in ("0-1h", ">=1h"):
                paired[metric][period].append(
                    float(np.nanmean(periods[period][metric]))
                )

    summary_metrics = {}
    lower_is_worse = {"finite_value_ratio"}
    for metric, periods in paired.items():
        early = np.asarray(periods["0-1h"], dtype=np.float64)
        late = np.asarray(periods[">=1h"], dtype=np.float64)
        difference = early - late
        early_worse = difference < 0 if metric in lower_is_worse else difference > 0
        summary_metrics[metric] = {
            "record_count": int(len(early)),
            "early_record_equal_median": float(np.nanmedian(early)),
            "late_record_equal_median": float(np.nanmedian(late)),
            "median_paired_difference_early_minus_late": float(
                np.nanmedian(difference)
            ),
            "fraction_records_early_worse": float(np.nanmean(early_worse)),
        }

    figure_path = output_dir / "ltaf_quality_comparison.png"
    figure, axes = plt.subplots(1, len(QUALITY_METRICS), figsize=(22, 4.5))
    for axis, metric in zip(axes, QUALITY_METRICS):
        early = np.asarray(paired[metric]["0-1h"])
        late = np.asarray(paired[metric][">=1h"])
        for first, second in zip(early, late):
            axis.plot([0, 1], [first, second], color="0.75", alpha=0.35, linewidth=0.7)
        axis.boxplot([early, late], positions=[0, 1], widths=0.4, showfliers=False)
        axis.set_xticks([0, 1], ["0-1h", ">=1h"])
        axis.set_title(metric.replace("_", "\n"), fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("LTAFDB label-free raw-signal quality: record-equal comparison")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    diagnostic = max_records is not None or windows_override is not None
    summary = {
        "dataset": "ltafdb",
        "dataset_version": config["dataset_version"],
        "comparison": "0-1h_vs_>=1h",
        "labels_accessed": False,
        "annotations_opened": False,
        "dynamic_patient_filtering": False,
        "diagnostic": diagnostic,
        "record_count": len(record_ids),
        "windows_per_period_per_record": sample_count,
        "metrics": summary_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "ltaf_quality_summary.json", summary)
    manifest = {
        **summary,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "revision_protocol_sha256": sha256_file(
            Path("EXPERIMENT_PLAN_REVISION_AFDB_SOURCE.md")
        ),
        "quality_csv_sha256": sha256_file(csv_path),
        "quality_figure_sha256": sha256_file(figure_path),
        "git": git_identity(),
    }
    _write_json(output_dir / "quality_run_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets/ltaf_clean1h_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--windows-per-period", type=int)
    args = parser.parse_args()
    if (args.max_records is not None or args.windows_per_period is not None) and (
        args.output_dir is None
    ):
        parser.error("diagnostic caps require --output-dir")
    result = audit_quality(
        args.config,
        output_override=args.output_dir,
        max_records=args.max_records,
        windows_override=args.windows_per_period,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
