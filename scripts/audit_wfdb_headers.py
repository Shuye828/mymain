#!/usr/bin/env python3
"""Read-only inventory of WFDB headers for the four AF datasets.

The script never loads signal samples, parses rhythm contents, or creates
windows. It calls ``wfdb.rdheader`` for every discovered ``.hea`` file and
writes a per-record CSV plus a summary JSON when output paths are requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DATASET_NAMES = ("ltafdb", "cpsc2021", "afdb", "shdb-af")
DEFAULT_ANNOTATION_EXTENSIONS = (".atr", ".ann", ".qrs")


@dataclass(frozen=True)
class HeaderAuditRow:
    """One WFDB header audit result."""

    dataset: str
    record_id: str
    header_path: str
    read_ok: bool
    error: str
    fs: float | None
    n_sig: int | None
    channel_names: list[str]
    sig_len: int | None
    duration_seconds: float | None
    annotation_exists: bool
    annotation_files: list[str]


def discover_headers(dataset_path: Path) -> list[Path]:
    """Return recursively discovered WFDB headers in deterministic order."""

    if not dataset_path.is_dir():
        return []
    return sorted(
        (path for path in dataset_path.rglob("*.hea") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def find_annotation_files(
    header_path: Path,
    annotation_extensions: Sequence[str] = DEFAULT_ANNOTATION_EXTENSIONS,
) -> list[Path]:
    """Find recognized annotation sidecars sharing a header's basename."""

    normalized = {
        extension if extension.startswith(".") else f".{extension}"
        for extension in annotation_extensions
    }
    return sorted(
        (
            candidate
            for candidate in header_path.parent.glob(f"{header_path.stem}.*")
            if candidate.is_file() and candidate.suffix.lower() in normalized
        ),
        key=lambda path: path.as_posix(),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def audit_dataset(
    dataset: str,
    dataset_path: Path,
    rdheader: Callable[[str], Any],
    annotation_extensions: Sequence[str] = DEFAULT_ANNOTATION_EXTENSIONS,
) -> list[HeaderAuditRow]:
    """Audit all headers under one dataset path without reading signal data."""

    rows: list[HeaderAuditRow] = []
    for header_path in discover_headers(dataset_path):
        record_path = header_path.with_suffix("")
        record_id = record_path.relative_to(dataset_path).as_posix()
        annotations = find_annotation_files(header_path, annotation_extensions)
        relative_annotations = [
            path.relative_to(dataset_path).as_posix() for path in annotations
        ]
        try:
            header = rdheader(str(record_path))
            fs = _optional_float(getattr(header, "fs", None))
            n_sig = _optional_int(getattr(header, "n_sig", None))
            sig_len = _optional_int(getattr(header, "sig_len", None))
            channel_names = [
                str(name) for name in (getattr(header, "sig_name", None) or [])
            ]
            duration = (
                float(sig_len) / fs
                if sig_len is not None and fs is not None and fs > 0
                else None
            )
            rows.append(
                HeaderAuditRow(
                    dataset=dataset,
                    record_id=record_id,
                    header_path=header_path.relative_to(dataset_path).as_posix(),
                    read_ok=True,
                    error="",
                    fs=fs,
                    n_sig=n_sig,
                    channel_names=channel_names,
                    sig_len=sig_len,
                    duration_seconds=duration,
                    annotation_exists=bool(annotations),
                    annotation_files=relative_annotations,
                )
            )
        except Exception as exc:  # Preserve the record and diagnostic in report.
            rows.append(
                HeaderAuditRow(
                    dataset=dataset,
                    record_id=record_id,
                    header_path=header_path.relative_to(dataset_path).as_posix(),
                    read_ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    fs=None,
                    n_sig=None,
                    channel_names=[],
                    sig_len=None,
                    duration_seconds=None,
                    annotation_exists=bool(annotations),
                    annotation_files=relative_annotations,
                )
            )
    return rows


def summarize_dataset(
    dataset: str, dataset_path: Path, rows: Sequence[HeaderAuditRow]
) -> dict[str, Any]:
    """Build a conservative completeness summary for one dataset."""

    readable = [row for row in rows if row.read_ok]
    valid = [
        row
        for row in readable
        if row.fs is not None
        and row.fs > 0
        and row.n_sig is not None
        and row.n_sig > 0
        and row.sig_len is not None
        and row.sig_len > 0
    ]
    annotation_count = sum(row.annotation_exists for row in rows)
    complete = (
        dataset_path.is_dir()
        and bool(rows)
        and len(readable) == len(rows)
        and len(valid) == len(rows)
        and annotation_count == len(rows)
    )
    return {
        "dataset": dataset,
        "path": str(dataset_path),
        "directory_exists": dataset_path.is_dir(),
        "header_count": len(rows),
        "readable_header_count": len(readable),
        "unreadable_header_count": len(rows) - len(readable),
        "valid_header_count": len(valid),
        "annotation_count": annotation_count,
        "missing_annotation_count": len(rows) - annotation_count,
        "sampling_rates": sorted({row.fs for row in readable if row.fs is not None}),
        "channel_counts": sorted(
            {row.n_sig for row in readable if row.n_sig is not None}
        ),
        "complete": complete,
    }


def write_csv(rows: Iterable[HeaderAuditRow], output_path: Path) -> None:
    """Write per-record audit results."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in HeaderAuditRow.__dataclass_fields__.values()]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = asdict(row)
            item["channel_names"] = json.dumps(item["channel_names"], ensure_ascii=False)
            item["annotation_files"] = json.dumps(
                item["annotation_files"], ensure_ascii=False
            )
            writer.writerow(item)


def write_json(summary: dict[str, Any], output_path: Path) -> None:
    """Write an aggregate audit summary."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _parse_dataset_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DATASET=PATH")
    dataset, raw_path = value.split("=", 1)
    if dataset not in DATASET_NAMES:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {dataset!r}; choose from {', '.join(DATASET_NAMES)}"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("dataset path cannot be empty")
    return dataset, Path(raw_path).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="root containing canonical dataset directory names (default: data/raw)",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset_override,
        default=[],
        metavar="DATASET=PATH",
        help="override one dataset path; may be repeated",
    )
    parser.add_argument(
        "--annotation-ext",
        action="append",
        default=[],
        help="recognized annotation suffix; may be repeated (default: atr, ann, qrs)",
    )
    parser.add_argument("--csv-out", type=Path, help="optional per-record CSV path")
    parser.add_argument("--json-out", type=Path, help="optional summary JSON path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 1 unless all four datasets are complete",
    )
    return parser


def _load_rdheader() -> Callable[[str], Any]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "wfdb is required; install dependencies with "
            "`python -m pip install -r requirements.txt`"
        ) from exc
    return wfdb.rdheader


def run(args: argparse.Namespace, rdheader: Callable[[str], Any]) -> dict[str, Any]:
    overrides = dict(args.dataset)
    extensions = args.annotation_ext or list(DEFAULT_ANNOTATION_EXTENSIONS)
    all_rows: list[HeaderAuditRow] = []
    summaries: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        dataset_path = overrides.get(dataset, args.data_root / dataset)
        rows = audit_dataset(dataset, dataset_path, rdheader, extensions)
        all_rows.extend(rows)
        summaries.append(summarize_dataset(dataset, dataset_path, rows))

    summary = {
        "datasets": summaries,
        "all_complete": all(item["complete"] for item in summaries),
        "total_headers": len(all_rows),
    }
    if args.csv_out:
        write_csv(all_rows, args.csv_out)
    if args.json_out:
        write_json(summary, args.json_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args, _load_rdheader())
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1 if args.strict and not summary["all_complete"] else 0


if __name__ == "__main__":
    sys.exit(main())
