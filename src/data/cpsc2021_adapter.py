"""Read-only adapter for the locally supplied CPSC2021 MATLAB v7.3 files."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from .rhythm_intervals import assert_complete_coverage, assert_interval_bounds
from .rhythm_mapping import RhythmMapping
from .schema import RecordMetadata, RhythmInterval


RECORD_PATTERN = re.compile(r"^data_(\d+)_(\d+)$")


def _decode_matlab_chars(dataset: h5py.Dataset) -> str:
    return "".join(chr(int(value)) for value in dataset[()].reshape(-1) if int(value))


def _is_matlab_empty(dataset: h5py.Dataset) -> bool:
    return bool(dataset.attrs.get("MATLAB_empty", 0))


class CPSC2021Adapter:
    """Use unprocessed lead arrays and official annotation-derived fields."""

    dataset = "cpsc2021"

    def __init__(self, *, root: Path, mapping: RhythmMapping) -> None:
        self.root = Path(root)
        self.mapping = mapping

    def list_records(self) -> list[str]:
        records = [
            path.stem
            for path in self.root.glob("data_*_*.mat")
            if RECORD_PATTERN.match(path.stem)
        ]
        return sorted(
            records,
            key=lambda value: tuple(int(part) for part in value.split("_")[1:]),
        )

    def _path(self, record_id: str) -> Path:
        if not RECORD_PATTERN.match(record_id):
            raise ValueError(f"invalid CPSC2021 record id: {record_id!r}")
        path = self.root / f"{record_id}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _subject_id(self, record_id: str) -> str:
        match = RECORD_PATTERN.match(record_id)
        if match is None:
            raise ValueError(f"invalid CPSC2021 record id: {record_id!r}")
        return match.group(1)

    @staticmethod
    def _vector_length(dataset: h5py.Dataset) -> int:
        if dataset.ndim not in (1, 2):
            raise ValueError(f"expected MATLAB vector, got shape {dataset.shape}")
        return int(max(dataset.shape))

    def read_metadata(self, record_id: str) -> RecordMetadata:
        path = self._path(record_id)
        with h5py.File(path, "r") as handle:
            record = handle["record"]
            fs = float(np.asarray(record["fs"][()]).reshape(-1)[0])
            lead1_len = self._vector_length(record["signal_lead1"])
            lead2_len = self._vector_length(record["signal_lead2"])
            if lead1_len != lead2_len:
                raise ValueError(
                    f"{record_id} raw lead lengths differ: {lead1_len} != {lead2_len}"
                )
            stored_name = _decode_matlab_chars(record["filename"])
            if stored_name and stored_name != record_id:
                raise ValueError(
                    f"filename mismatch: path={record_id!r}, stored={stored_name!r}"
                )
        return RecordMetadata(
            dataset=self.dataset,
            record_id=record_id,
            subject_id=self._subject_id(record_id),
            source_path=path.relative_to(self.root).as_posix(),
            fs=fs,
            channel_names=("I", "II"),
            signal_length=lead1_len,
            has_signal=True,
            has_annotation=True,
            annotation_source=f"{record_id}.mat:record",
        )

    @staticmethod
    def _read_vector_slice(
        dataset: h5py.Dataset, start_sample: int, end_sample: int
    ) -> np.ndarray:
        if dataset.ndim == 1:
            return np.asarray(dataset[start_sample:end_sample])
        if dataset.shape[0] == 1:
            return np.asarray(dataset[0, start_sample:end_sample])
        if dataset.shape[1] == 1:
            return np.asarray(dataset[start_sample:end_sample, 0])
        raise ValueError(f"expected MATLAB vector, got shape {dataset.shape}")

    def read_signal(
        self, record_id: str, start_sample: int = 0, end_sample: int | None = None
    ) -> np.ndarray:
        """Read only signal_lead1/2; never use the stored processed copies."""

        metadata = self.read_metadata(record_id)
        end = metadata.signal_length if end_sample is None else int(end_sample)
        if start_sample < 0 or end <= start_sample or end > metadata.signal_length:
            raise ValueError("invalid signal sample bounds")
        with h5py.File(self._path(record_id), "r") as handle:
            record = handle["record"]
            lead1 = self._read_vector_slice(
                record["signal_lead1"], start_sample, end
            )
            lead2 = self._read_vector_slice(
                record["signal_lead2"], start_sample, end
            )
        return np.stack((lead1, lead2), axis=0)

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _af_sample_ranges(
        self, record: h5py.Group, signal_length: int
    ) -> list[tuple[int, int]]:
        starts_ds = record["AF_startPoints_byOfficalRwave"]
        ends_ds = record["AF_endPoints_byOfficalRwave"]
        if _is_matlab_empty(starts_ds) or _is_matlab_empty(ends_ds):
            return []
        starts = np.rint(np.asarray(starts_ds[()]).reshape(-1)).astype(int)
        ends = np.rint(np.asarray(ends_ds[()]).reshape(-1)).astype(int)
        rpeaks = np.rint(
            np.asarray(record["offical_RwavePos"][()]).reshape(-1)
        ).astype(int)
        if len(starts) != len(ends):
            raise ValueError("CPSC AF start/end counts differ")
        if len(rpeaks) == 0:
            raise ValueError("CPSC record has no official R peaks")

        ranges: list[tuple[int, int]] = []
        for start_index, end_index in zip(starts, ends):
            if start_index < 0 or start_index > len(rpeaks):
                raise ValueError(f"invalid CPSC AF start index {start_index}")
            if end_index < 1 or end_index > len(rpeaks):
                raise ValueError(f"invalid CPSC AF end index {end_index}")
            # Local MATLAB code treats zero as a boundary before the first beat.
            start = 0 if start_index == 0 else int(rpeaks[start_index - 1] - 1)
            # An end at/after the final indexed beat means AF continues to EOF.
            end = (
                signal_length
                if end_index >= len(rpeaks)
                else int(rpeaks[end_index - 1])
            )
            start = max(0, min(start, signal_length))
            end = max(0, min(end, signal_length))
            if end <= start:
                raise ValueError(f"invalid CPSC AF sample range {start}:{end}")
            ranges.append((start, end))
        return self._merge_ranges(ranges)

    def read_rhythm_intervals(self, record_id: str) -> list[RhythmInterval]:
        metadata = self.read_metadata(record_id)
        source = metadata.annotation_source or f"{record_id}.mat:record"
        with h5py.File(self._path(record_id), "r") as handle:
            record = handle["record"]
            label = _decode_matlab_chars(record["label"])
            if label == "Normal":
                intervals = [
                    RhythmInterval(
                        0,
                        metadata.signal_length,
                        "Normal",
                        self.mapping.action_for(self.dataset, "Normal"),
                        source,
                    )
                ]
            elif label == "AEf":
                intervals = [
                    RhythmInterval(
                        0,
                        metadata.signal_length,
                        "AEf",
                        self.mapping.action_for(self.dataset, "AEf"),
                        source,
                    )
                ]
            elif label == "AFp":
                af_ranges = self._af_sample_ranges(record, metadata.signal_length)
                if not af_ranges:
                    raise ValueError(f"{record_id} is AFp but has no AF boundaries")
                intervals = []
                cursor = 0
                for start, end in af_ranges:
                    if cursor < start:
                        intervals.append(
                            RhythmInterval(
                                cursor,
                                start,
                                "AFp_nonAF",
                                self.mapping.action_for(
                                    self.dataset, "AFp_nonAF"
                                ),
                                source,
                            )
                        )
                    intervals.append(
                        RhythmInterval(
                            start,
                            end,
                            "AFp_AF",
                            self.mapping.action_for(self.dataset, "AFp_AF"),
                            source,
                        )
                    )
                    cursor = end
                if cursor < metadata.signal_length:
                    intervals.append(
                        RhythmInterval(
                            cursor,
                            metadata.signal_length,
                            "AFp_nonAF",
                            self.mapping.action_for(self.dataset, "AFp_nonAF"),
                            source,
                        )
                    )
            else:
                intervals = [
                    RhythmInterval(
                        0,
                        metadata.signal_length,
                        label or "__UNKNOWN_CPSC_LABEL__",
                        "exclude",
                        source,
                    )
                ]
        assert_interval_bounds(intervals, metadata.signal_length)
        assert_complete_coverage(intervals, metadata.signal_length)
        return intervals
