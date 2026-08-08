from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.rhythm_mapping import load_rhythm_mapping
from src.data.wfdb_adapter import WFDBDatasetAdapter, load_shdb_subject_maps


class FakeWFDB:
    def __init__(self, *, n_sig: int = 2) -> None:
        self.n_sig = n_sig

    def rdheader(self, _: str) -> SimpleNamespace:
        return SimpleNamespace(
            fs=250,
            n_sig=self.n_sig,
            sig_len=100,
            sig_name=["ECG1", "ECG2"] if self.n_sig else [],
        )

    def rdann(self, _: str, extension: str) -> SimpleNamespace:
        assert extension == "atr"
        return SimpleNamespace(
            sample=np.asarray([10, 50]),
            aux_note=["(N", "(AFIB"],
        )

    def rdrecord(
        self, _: str, *, sampfrom: int, sampto: int, physical: bool
    ) -> SimpleNamespace:
        assert physical
        return SimpleNamespace(
            p_signal=np.arange((sampto - sampfrom) * 2).reshape(-1, 2)
        )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_wfdb_adapter_reads_metadata_signal_and_intervals(tmp_path: Path) -> None:
    for suffix in (".hea", ".dat", ".atr"):
        _touch(tmp_path / f"r{suffix}")
    adapter = WFDBDatasetAdapter(
        dataset="afdb",
        root=tmp_path,
        mapping=load_rhythm_mapping(),
        wfdb_module=FakeWFDB(),
    )

    metadata = adapter.read_metadata("r")
    signal = adapter.read_signal("r", 2, 7)
    intervals = adapter.read_rhythm_intervals("r")

    assert metadata.subject_id == "r"
    assert metadata.channel_names == ("ECG1", "ECG2")
    assert signal.shape == (2, 5)
    assert [item.action for item in intervals] == ["exclude", "nonaf", "af"]


def test_missing_annotation_returns_no_intervals(tmp_path: Path) -> None:
    for suffix in (".hea", ".dat"):
        _touch(tmp_path / f"r{suffix}")
    adapter = WFDBDatasetAdapter(
        dataset="ltafdb",
        root=tmp_path,
        mapping=load_rhythm_mapping(),
        wfdb_module=FakeWFDB(),
    )

    assert not adapter.read_metadata("r").has_annotation
    assert adapter.read_rhythm_intervals("r") == []


def test_annotation_only_record_rejects_signal_read(tmp_path: Path) -> None:
    for suffix in (".hea", ".atr"):
        _touch(tmp_path / f"r{suffix}")
    adapter = WFDBDatasetAdapter(
        dataset="afdb",
        root=tmp_path,
        mapping=load_rhythm_mapping(),
        wfdb_module=FakeWFDB(n_sig=0),
    )

    assert not adapter.read_metadata("r").has_signal
    try:
        adapter.read_signal("r", 0, 10)
    except ValueError as exc:
        assert "has no signal" in str(exc)
    else:
        raise AssertionError("annotation-only record unexpectedly returned signal")


def test_shdb_subject_mapping_uses_subject_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "AdditionalData.csv"
    csv_path.write_text(
        "Data_ID,Subject_ID,Annotated\n1,patient-a,True\n2,patient-a,False\n",
        encoding="utf-8",
    )

    subjects, annotations = load_shdb_subject_maps(csv_path)

    assert subjects == {"001": "patient-a", "002": "patient-a"}
    assert annotations == {"001": True, "002": False}
