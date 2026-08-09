import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.data.ecg_dataset as dataset_module
from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.representation.source_export import export_source_direction
from src.training.reproducibility import sha256_file


def _write_index(path: Path) -> None:
    fields = [
        "dataset",
        "record_id",
        "subject_id",
        "start_sample",
        "end_sample",
        "fs_original",
        "binary_label",
        "rhythm_label",
        "source_split",
        "target_split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in (0, 1):
            for index in range(2):
                writer.writerow(
                    {
                        "dataset": "cpsc2021",
                        "record_id": f"record_{label}_{index}",
                        "subject_id": f"subject_{label}",
                        "start_sample": index * 2000,
                        "end_sample": (index + 1) * 2000,
                        "fs_original": 200,
                        "binary_label": label,
                        "rhythm_label": "fixture",
                        "source_split": "train",
                        "target_split": "adaptation",
                    }
                )


def test_source_export_preserves_metadata_and_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeAdapter:
        def read_signal(self, record_id: str, start: int, end: int) -> np.ndarray:
            label = float(record_id.split("_")[1])
            record_index = float(record_id.split("_")[2])
            time = np.arange(end - start, dtype=np.float64) / 200.0
            return np.stack(
                [
                    np.sin(
                        2 * np.pi * (4 + label + 0.1 * record_index) * time
                    ),
                    np.cos(
                        2 * np.pi * (6 + label + 0.1 * record_index) * time
                    ),
                ]
            )

    monkeypatch.setattr(
        dataset_module, "create_adapter", lambda dataset, data_root: FakeAdapter()
    )
    index_path = tmp_path / "windows.csv"
    source_config_path = tmp_path / "source.json"
    checkpoint_path = tmp_path / "best.pt"
    output_dir = tmp_path / "output"
    _write_index(index_path)
    source_config = {
        "role": "source",
        "dataset": "cpsc2021",
        "data_root": str(tmp_path),
        "index_path": str(index_path),
        "model": {
            "dim": 16,
            "max_channel": 2,
            "num_heads": 4,
            "num_layers": 1,
            "patch_size": 8,
            "num_classes": 2,
        },
        "training": {
            "seed": 42,
            "max_windows_per_subject_per_class": 10,
        },
    }
    source_config_path.write_text(json.dumps(source_config), encoding="utf-8")
    model = SourceMedTSTTT(**source_config["model"])
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "provenance": {
                "config": source_config,
                "index_sha256": sha256_file(index_path),
            },
        },
        checkpoint_path,
    )
    config = {
        "role": "source_direction",
        "dataset": "cpsc2021",
        "source_config": str(source_config_path),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "representation": {"kind": "backbone_l2", "expected_dim": 16},
        "export": {
            "batch_size": 2,
            "num_workers": 0,
            "progress_every_batches": 1,
        },
    }

    result = export_source_direction(config, device_request="cpu")

    assert result["window_count"] == 4
    assert result["feature_dim"] == 16
    assert result["checkpoint_epoch"] == 3
    assert result["feature_norm_max_abs_error"] < 1e-5
    archive = np.load(output_dir / "source_train_features.npz")
    assert archive["features"].shape == (4, 16)
    assert set(archive["labels"].tolist()) == {0, 1}
    assert len(archive["subject_id"]) == 4
    assert len(archive["record_id"]) == 4
    assert len(archive["window_start"]) == 4
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["source_split"] == "train"
    assert manifest["index_sha256"] == sha256_file(index_path)
    assert manifest["checkpoint_sha256"] == sha256_file(checkpoint_path)
    direction = result["direction"]
    assert direction["source_fixed_threshold_rule"] == (
        "source_prototype_projection_midpoint"
    )
    assert direction["pooled_separation"] > 0
    assert direction["nonaf_projection_mean"] < direction["source_fixed_threshold"]
    assert direction["source_fixed_threshold"] < direction["af_projection_mean"]


def test_diagnostic_export_requires_separate_output() -> None:
    with pytest.raises(ValueError, match="explicit output override"):
        export_source_direction(
            {
                "role": "source_direction",
                "representation": {"kind": "backbone_l2"},
            },
            device_request="cpu",
            max_batches=1,
        )
