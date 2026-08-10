import csv

from src.data.rhythm_mapping import RhythmMapping
from src.data.schema import RecordMetadata, RhythmInterval
from src.data.splits import SubjectSplit
from src.data.window_index import classify_grid_windows, index_dataset


def _interval(start: int, end: int, token: str, action: str) -> RhythmInterval:
    return RhythmInterval(start, end, token, action, "fixture.atr")


def test_grid_windows_accept_only_full_allowed_interval_containment() -> None:
    intervals = [
        _interval(0, 15, "(N", "nonaf"),
        _interval(15, 35, "(AFIB", "af"),
        _interval(35, 50, "(AFL", "exclude"),
    ]

    decisions = list(
        classify_grid_windows(
            signal_length=50,
            intervals=intervals,
            window_samples=10,
            stride_samples=10,
        )
    )

    assert [
        (
            item.start_sample,
            item.end_sample,
            item.reason,
            None if item.interval is None else item.interval.action,
        )
        for item in decisions
    ] == [
        (0, 10, "accepted", "nonaf"),
        (10, 20, "transition", None),
        (20, 30, "accepted", "af"),
        (30, 40, "transition", None),
        (40, 50, "excluded_rhythm", "exclude"),
    ]


def test_short_record_produces_no_windows() -> None:
    decisions = list(
        classify_grid_windows(
            signal_length=9,
            intervals=[_interval(0, 9, "(N", "nonaf")],
            window_samples=10,
            stride_samples=10,
        )
    )

    assert decisions == []


def test_minimum_start_excludes_prior_and_crossing_windows() -> None:
    decisions = list(
        classify_grid_windows(
            signal_length=60,
            intervals=[_interval(0, 60, "(N", "nonaf")],
            window_samples=10,
            stride_samples=10,
            minimum_start_sample=35,
        )
    )

    assert [(item.start_sample, item.reason) for item in decisions] == [
        (0, "before_minimum_start"),
        (10, "before_minimum_start"),
        (20, "before_minimum_start"),
        (30, "before_minimum_start"),
        (40, "accepted"),
        (50, "accepted"),
    ]


def test_minimum_start_defaults_to_historical_zero_behavior() -> None:
    kwargs = {
        "signal_length": 30,
        "intervals": [_interval(0, 30, "(N", "nonaf")],
        "window_samples": 10,
        "stride_samples": 10,
    }

    historical = list(classify_grid_windows(**kwargs))
    explicit_zero = list(classify_grid_windows(**kwargs, minimum_start_sample=0))

    assert historical == explicit_zero


def test_index_dataset_applies_minimum_start_and_records_statistics(tmp_path) -> None:
    class Adapter:
        dataset = "ltafdb"

        @staticmethod
        def list_records() -> list[str]:
            return ["00"]

        @staticmethod
        def read_metadata(record_id: str) -> RecordMetadata:
            return RecordMetadata(
                dataset="ltafdb",
                record_id=record_id,
                subject_id=record_id,
                source_path=f"{record_id}.hea",
                fs=1.0,
                channel_names=("ECG1", "ECG2"),
                signal_length=50,
                has_signal=True,
                has_annotation=True,
                annotation_source=f"{record_id}.atr",
            )

        @staticmethod
        def read_rhythm_intervals(record_id: str) -> list[RhythmInterval]:
            return [_interval(0, 50, "(N", "nonaf")]

    output = tmp_path / "clean.csv"
    result = index_dataset(
        adapter=Adapter(),
        subject_splits={
            "00": SubjectSplit(
                dataset="ltafdb",
                subject_id="00",
                source_split="train",
                target_split="adaptation",
                eligible_record_count=1,
                seed=42,
                split_version="fixture_split_v1",
            )
        },
        output_path=output,
        mapping=RhythmMapping("fixture_mapping_v1", {"ltafdb": {"(N": "nonaf"}}),
        window_config={
            "duration_seconds": 10,
            "stride_seconds": 10,
            "version": "fixture_clean_v1",
            "cpsc_boundary_version": "not_applicable",
        },
        minimum_start_seconds=20,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["start_sample"]) for row in rows] == [20, 30, 40]
    assert {row["window_version"] for row in rows} == {"fixture_clean_v1"}
    assert result["accepted_windows"] == 3
    assert result["statistics"]["window_before_minimum_start"] == 2
