from src.data.schema import RhythmInterval
from src.data.window_index import classify_grid_windows


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
