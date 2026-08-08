"""Load and query the reviewed strict AF rhythm mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .schema import LabelAction


DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "datasets" / "rhythm_mapping.json"
)
VALID_ACTIONS = {"af", "nonaf", "exclude"}


@dataclass(frozen=True)
class RhythmMapping:
    version: str
    datasets: Mapping[str, Mapping[str, LabelAction]]

    def action_for(self, dataset: str, raw_token: str) -> LabelAction:
        """Return a conservative action; unknown tokens are always excluded."""

        action = self.datasets.get(dataset, {}).get(raw_token, "exclude")
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid rhythm action {action!r}")
        return action


def load_rhythm_mapping(path: Path = DEFAULT_MAPPING_PATH) -> RhythmMapping:
    """Load the auditable JSON mapping used by adapters and inventories."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("rhythm mapping requires a 'datasets' object")
    for dataset, mapping in datasets.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"mapping for {dataset!r} must be an object")
        invalid = set(mapping.values()) - VALID_ACTIONS
        if invalid:
            raise ValueError(f"mapping for {dataset!r} has invalid actions: {invalid}")
    return RhythmMapping(version=str(payload["version"]), datasets=datasets)
