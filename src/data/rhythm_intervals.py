"""Pure conversions from rhythm-change markers to half-open intervals."""

from __future__ import annotations

from collections.abc import Iterable

from .rhythm_mapping import RhythmMapping
from .schema import RhythmInterval


UNANNOTATED_TOKEN = "__UNANNOTATED__"


def wfdb_markers_to_intervals(
    *,
    dataset: str,
    signal_length: int,
    marker_samples: Iterable[int],
    marker_tokens: Iterable[str],
    annotation_source: str,
    mapping: RhythmMapping,
) -> list[RhythmInterval]:
    """Convert WFDB rhythm-change events into complete record coverage.

    Events without a token beginning with ``("`` must be filtered by the
    caller. Samples before the first rhythm marker are explicitly marked
    unannotated rather than inheriting a label.
    """

    if signal_length <= 0:
        return []

    events: dict[int, str] = {}
    for sample, token in zip(marker_samples, marker_tokens):
        sample_int = int(sample)
        token_text = str(token).strip()
        if not token_text.startswith("("):
            continue
        if 0 <= sample_int < signal_length:
            events[sample_int] = token_text

    ordered = sorted(events.items())
    if not ordered:
        return [
            RhythmInterval(
                0,
                signal_length,
                UNANNOTATED_TOKEN,
                mapping.action_for(dataset, UNANNOTATED_TOKEN),
                annotation_source,
            )
        ]

    intervals: list[RhythmInterval] = []
    first_sample = ordered[0][0]
    if first_sample > 0:
        intervals.append(
            RhythmInterval(
                0,
                first_sample,
                UNANNOTATED_TOKEN,
                mapping.action_for(dataset, UNANNOTATED_TOKEN),
                annotation_source,
            )
        )

    for index, (start, token) in enumerate(ordered):
        end = ordered[index + 1][0] if index + 1 < len(ordered) else signal_length
        if end <= start:
            continue
        intervals.append(
            RhythmInterval(
                start,
                end,
                token,
                mapping.action_for(dataset, token),
                annotation_source,
            )
        )
    return intervals


def assert_interval_bounds(
    intervals: Iterable[RhythmInterval], signal_length: int
) -> None:
    """Validate sorted, non-overlapping intervals within a record."""

    previous_end = 0
    for interval in intervals:
        if interval.start_sample < previous_end:
            raise ValueError("rhythm intervals overlap or are out of order")
        if interval.end_sample > signal_length:
            raise ValueError("rhythm interval exceeds signal length")
        previous_end = interval.end_sample


def assert_complete_coverage(
    intervals: Iterable[RhythmInterval], signal_length: int
) -> None:
    """Require adjacent intervals to cover every sample exactly once."""

    items = list(intervals)
    if signal_length == 0 and not items:
        return
    if not items or items[0].start_sample != 0:
        raise ValueError("rhythm intervals do not start at sample zero")
    for previous, current in zip(items, items[1:]):
        if previous.end_sample != current.start_sample:
            raise ValueError("rhythm intervals contain a gap or overlap")
    if items[-1].end_sample != signal_length:
        raise ValueError("rhythm intervals do not reach the signal end")
