from pathlib import Path

import h5py
import numpy as np

from src.data.cpsc2021_adapter import CPSC2021Adapter
from src.data.rhythm_mapping import load_rhythm_mapping


def _chars(value: str) -> np.ndarray:
    return np.asarray([ord(char) for char in value], dtype=np.uint16).reshape(-1, 1)


def _write_cpsc_record(
    path: Path,
    *,
    label: str,
    length: int,
    starts: np.ndarray | None,
    ends: np.ndarray | None,
) -> None:
    with h5py.File(path, "w") as handle:
        record = handle.create_group("record")
        record.create_dataset("fs", data=np.asarray([[200.0]]))
        record.create_dataset("filename", data=_chars(path.stem))
        record.create_dataset("label", data=_chars(label))
        record.create_dataset("signal_lead1", data=np.arange(length).reshape(1, -1))
        record.create_dataset(
            "signal_lead2", data=(np.arange(length) + 100).reshape(1, -1)
        )
        record.create_dataset(
            "signal_lead1_processed", data=np.full((1, length), -1)
        )
        record.create_dataset(
            "signal_lead2_processed", data=np.full((1, length), -2)
        )
        record.create_dataset(
            "offical_RwavePos", data=np.asarray([[10, 20, 30, 40]])
        )
        if starts is None:
            start_ds = record.create_dataset(
                "AF_startPoints_byOfficalRwave",
                data=np.asarray([0, 1], dtype=np.uint64),
            )
            end_ds = record.create_dataset(
                "AF_endPoints_byOfficalRwave",
                data=np.asarray([0, 1], dtype=np.uint64),
            )
            start_ds.attrs["MATLAB_empty"] = 1
            end_ds.attrs["MATLAB_empty"] = 1
        else:
            record.create_dataset(
                "AF_startPoints_byOfficalRwave", data=starts.reshape(1, -1)
            )
            record.create_dataset(
                "AF_endPoints_byOfficalRwave", data=ends.reshape(1, -1)
            )


def test_cpsc_adapter_reads_only_unprocessed_leads_and_subject(tmp_path: Path) -> None:
    path = tmp_path / "data_7_1.mat"
    _write_cpsc_record(
        path,
        label="AFp",
        length=50,
        starts=np.asarray([0]),
        ends=np.asarray([3]),
    )
    adapter = CPSC2021Adapter(root=tmp_path, mapping=load_rhythm_mapping())

    metadata = adapter.read_metadata("data_7_1")
    signal = adapter.read_signal("data_7_1", 2, 5)
    intervals = adapter.read_rhythm_intervals("data_7_1")

    assert metadata.subject_id == "7"
    assert metadata.channel_names == ("I", "II")
    assert signal.tolist() == [[2, 3, 4], [102, 103, 104]]
    assert [
        (item.start_sample, item.end_sample, item.raw_token, item.action)
        for item in intervals
    ] == [
        (0, 30, "AFp_AF", "af"),
        (30, 50, "AFp_nonAF", "nonaf"),
    ]


def test_cpsc_normal_short_record_is_nonaf(tmp_path: Path) -> None:
    path = tmp_path / "data_8_1.mat"
    _write_cpsc_record(
        path, label="Normal", length=5, starts=None, ends=None
    )
    adapter = CPSC2021Adapter(root=tmp_path, mapping=load_rhythm_mapping())

    intervals = adapter.read_rhythm_intervals("data_8_1")

    assert len(intervals) == 1
    assert (intervals[0].start_sample, intervals[0].end_sample) == (0, 5)
    assert intervals[0].action == "nonaf"
