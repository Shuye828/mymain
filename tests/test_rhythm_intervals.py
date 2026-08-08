from pathlib import Path

from src.data.rhythm_intervals import UNANNOTATED_TOKEN, wfdb_markers_to_intervals
from src.data.rhythm_mapping import load_rhythm_mapping


def test_wfdb_markers_preserve_transition_and_unannotated_prefix() -> None:
    mapping = load_rhythm_mapping()
    intervals = wfdb_markers_to_intervals(
        dataset="afdb",
        signal_length=100,
        marker_samples=[10, 50],
        marker_tokens=["(N", "(AFIB"],
        annotation_source="record.atr",
        mapping=mapping,
    )

    assert [
        (item.start_sample, item.end_sample, item.raw_token, item.action)
        for item in intervals
    ] == [
        (0, 10, UNANNOTATED_TOKEN, "exclude"),
        (10, 50, "(N", "nonaf"),
        (50, 100, "(AFIB", "af"),
    ]


def test_unknown_rhythm_is_excluded_by_default() -> None:
    mapping = load_rhythm_mapping()
    intervals = wfdb_markers_to_intervals(
        dataset="afdb",
        signal_length=20,
        marker_samples=[0],
        marker_tokens=["(UNREVIEWED"],
        annotation_source="record.atr",
        mapping=mapping,
    )

    assert intervals[0].raw_token == "(UNREVIEWED"
    assert intervals[0].action == "exclude"
