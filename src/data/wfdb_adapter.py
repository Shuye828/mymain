"""Read-only adapter for WFDB-formatted AF datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .rhythm_intervals import (
    assert_complete_coverage,
    assert_interval_bounds,
    wfdb_markers_to_intervals,
)
from .rhythm_mapping import RhythmMapping
from .schema import RecordMetadata, RhythmInterval


class WFDBDatasetAdapter:
    """Expose metadata, raw decoded signals, and rhythm intervals."""

    def __init__(
        self,
        *,
        dataset: str,
        root: Path,
        mapping: RhythmMapping,
        wfdb_module: Any | None = None,
        subject_map: Mapping[str, str] | None = None,
        annotation_map: Mapping[str, bool] | None = None,
    ) -> None:
        self.dataset = dataset
        self.root = Path(root)
        self.mapping = mapping
        self._wfdb = wfdb_module
        self.subject_map = dict(subject_map or {})
        self.annotation_map = dict(annotation_map or {})

    @property
    def wfdb(self) -> Any:
        if self._wfdb is None:
            try:
                import wfdb
            except ImportError as exc:
                raise RuntimeError(
                    "wfdb is required for WFDB datasets; install requirements.txt"
                ) from exc
            self._wfdb = wfdb
        return self._wfdb

    def list_records(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.hea"))

    def _record_path(self, record_id: str) -> Path:
        path = self.root / record_id
        if not path.with_suffix(".hea").is_file():
            raise FileNotFoundError(f"missing WFDB header: {path}.hea")
        return path

    def read_metadata(self, record_id: str) -> RecordMetadata:
        path = self._record_path(record_id)
        header = self.wfdb.rdheader(str(path))
        n_sig = int(header.n_sig)
        signal_length = int(header.sig_len)
        has_signal = n_sig > 0 and path.with_suffix(".dat").is_file()
        annotation_exists = path.with_suffix(".atr").is_file()
        if record_id in self.annotation_map:
            annotation_exists = annotation_exists and self.annotation_map[record_id]
        return RecordMetadata(
            dataset=self.dataset,
            record_id=record_id,
            subject_id=self.subject_map.get(record_id, record_id),
            source_path=path.with_suffix(".hea").relative_to(self.root).as_posix(),
            fs=float(header.fs),
            channel_names=tuple(str(name) for name in (header.sig_name or [])),
            signal_length=signal_length,
            has_signal=has_signal,
            has_annotation=annotation_exists,
            annotation_source=f"{record_id}.atr" if annotation_exists else None,
        )

    def read_signal(
        self, record_id: str, start_sample: int = 0, end_sample: int | None = None
    ) -> np.ndarray:
        """Read physical units without filtering, resampling, or normalization."""

        metadata = self.read_metadata(record_id)
        if not metadata.has_signal:
            raise ValueError(f"{self.dataset}/{record_id} has no signal")
        end = metadata.signal_length if end_sample is None else int(end_sample)
        if start_sample < 0 or end <= start_sample or end > metadata.signal_length:
            raise ValueError("invalid signal sample bounds")
        record = self.wfdb.rdrecord(
            str(self.root / record_id),
            sampfrom=int(start_sample),
            sampto=end,
            physical=True,
        )
        signal = np.asarray(record.p_signal)
        if signal.ndim != 2:
            raise ValueError(f"unexpected WFDB signal shape {signal.shape}")
        return signal.T

    def read_rhythm_intervals(self, record_id: str) -> list[RhythmInterval]:
        metadata = self.read_metadata(record_id)
        if not metadata.has_annotation:
            return []
        annotation = self.wfdb.rdann(str(self.root / record_id), "atr")
        intervals = wfdb_markers_to_intervals(
            dataset=self.dataset,
            signal_length=metadata.signal_length,
            marker_samples=annotation.sample,
            marker_tokens=annotation.aux_note,
            annotation_source=f"{record_id}.atr",
            mapping=self.mapping,
        )
        assert_interval_bounds(intervals, metadata.signal_length)
        assert_complete_coverage(intervals, metadata.signal_length)
        return intervals


def load_shdb_subject_maps(
    csv_path: Path,
) -> tuple[dict[str, str], dict[str, bool]]:
    """Load record-to-subject and annotation flags from AdditionalData.csv."""

    subject_map: dict[str, str] = {}
    annotation_map: dict[str, bool] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            record_id = str(row["Data_ID"]).zfill(3)
            subject_map[record_id] = str(row["Subject_ID"])
            annotation_map[record_id] = str(row["Annotated"]).strip().lower() == "true"
    return subject_map, annotation_map
