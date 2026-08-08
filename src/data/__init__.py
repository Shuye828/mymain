"""Dataset adapters and rhythm annotation contracts."""

from .registry import create_adapter
from .schema import LabelAction, RecordMetadata, RhythmInterval

__all__ = [
    "LabelAction",
    "RecordMetadata",
    "RhythmInterval",
    "create_adapter",
]
