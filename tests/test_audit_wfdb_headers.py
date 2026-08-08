from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_wfdb_headers import (
    DATASET_NAMES,
    audit_dataset,
    discover_headers,
    run,
    summarize_dataset,
)


def _touch(path: Path, contents: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_audit_dataset_reads_headers_and_annotations(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ltafdb"
    _touch(dataset_path / "nested" / "record.hea")
    _touch(dataset_path / "nested" / "record.atr")
    calls: list[str] = []

    def fake_rdheader(record_path: str) -> SimpleNamespace:
        calls.append(record_path)
        return SimpleNamespace(fs=128, n_sig=2, sig_name=["ECG1", "ECG2"], sig_len=1280)

    rows = audit_dataset("ltafdb", dataset_path, fake_rdheader)

    assert len(rows) == 1
    assert calls == [str(dataset_path / "nested" / "record")]
    assert rows[0].record_id == "nested/record"
    assert rows[0].duration_seconds == 10.0
    assert rows[0].annotation_exists
    assert summarize_dataset("ltafdb", dataset_path, rows)["complete"]


def test_unreadable_header_is_reported_not_dropped(tmp_path: Path) -> None:
    dataset_path = tmp_path / "afdb"
    _touch(dataset_path / "bad.hea")
    _touch(dataset_path / "bad.atr")

    def failing_rdheader(_: str) -> None:
        raise ValueError("malformed header")

    rows = audit_dataset("afdb", dataset_path, failing_rdheader)
    summary = summarize_dataset("afdb", dataset_path, rows)

    assert len(rows) == 1
    assert not rows[0].read_ok
    assert rows[0].error == "ValueError: malformed header"
    assert summary["unreadable_header_count"] == 1
    assert not summary["complete"]


def test_missing_dataset_has_no_headers_and_is_incomplete(tmp_path: Path) -> None:
    missing = tmp_path / "shdb-af"

    assert discover_headers(missing) == []
    summary = summarize_dataset("shdb-af", missing, [])
    assert not summary["directory_exists"]
    assert summary["header_count"] == 0
    assert not summary["complete"]


def test_run_writes_csv_and_json_for_all_datasets(
    tmp_path: Path, capsys
) -> None:
    data_root = tmp_path / "raw"
    for dataset in DATASET_NAMES:
        _touch(data_root / dataset / "r.hea")
        _touch(data_root / dataset / "r.atr")

    def fake_rdheader(_: str) -> SimpleNamespace:
        return SimpleNamespace(fs=200, n_sig=2, sig_name=["I", "II"], sig_len=2000)

    csv_out = tmp_path / "reports" / "audit.csv"
    json_out = tmp_path / "reports" / "audit.json"
    args = Namespace(
        data_root=data_root,
        dataset=[],
        annotation_ext=[],
        csv_out=csv_out,
        json_out=json_out,
        strict=True,
    )

    summary = run(args, fake_rdheader)
    captured = capsys.readouterr()

    assert summary["all_complete"]
    assert summary["total_headers"] == 4
    assert csv_out.read_text(encoding="utf-8").count("\n") == 5
    assert json.loads(json_out.read_text(encoding="utf-8"))["all_complete"]
    assert '"all_complete": true' in captured.out
