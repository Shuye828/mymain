"""Shared immutable data structures for source-format ECG adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LabelAction = Literal["af", "nonaf", "exclude"]


@dataclass(frozen=True)
class RecordMetadata:
    """Metadata required before any ECG window is created."""

    dataset: str
    record_id: str
    subject_id: str
    source_path: str
    fs: float
    channel_names: tuple[str, ...]
    signal_length: int
    has_signal: bool
    has_annotation: bool
    annotation_source: str | None

    @property
    def duration_seconds(self) -> float:
        return self.signal_length / self.fs if self.fs > 0 else 0.0


@dataclass(frozen=True)
class RhythmInterval:
    """Half-open source-rate rhythm interval with its unmodified raw token."""

    start_sample: int
    end_sample: int
    raw_token: str
    action: LabelAction
    annotation_source: str

    def __post_init__(self) -> None:
        if self.start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")

    @property
    def length(self) -> int:
        return self.end_sample - self.start_sample
