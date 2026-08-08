"""Canonical construction of the four dataset adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cpsc2021_adapter import CPSC2021Adapter
from .rhythm_mapping import RhythmMapping, load_rhythm_mapping
from .wfdb_adapter import WFDBDatasetAdapter, load_shdb_subject_maps


DATASET_NAMES = ("ltafdb", "cpsc2021", "afdb", "shdb-af")


def create_adapter(
    dataset: str,
    *,
    data_root: Path = Path("data/raw"),
    mapping: RhythmMapping | None = None,
    wfdb_module: Any | None = None,
):
    """Create the canonical source-format adapter for one dataset."""

    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {dataset!r}")
    mapping = mapping or load_rhythm_mapping()
    root = Path(data_root) / dataset
    if dataset == "cpsc2021":
        return CPSC2021Adapter(root=root, mapping=mapping)
    if dataset == "shdb-af":
        subject_map, annotation_map = load_shdb_subject_maps(
            root / "AdditionalData.csv"
        )
        return WFDBDatasetAdapter(
            dataset=dataset,
            root=root,
            mapping=mapping,
            wfdb_module=wfdb_module,
            subject_map=subject_map,
            annotation_map=annotation_map,
        )
    return WFDBDatasetAdapter(
        dataset=dataset,
        root=root,
        mapping=mapping,
        wfdb_module=wfdb_module,
    )
