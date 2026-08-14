import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.training.axis_alignment import (
    axis_alignment_loss,
    binary_head_direction,
    extract_source_axis,
    lambda_slug,
    unit_vector,
    validate_m2_config,
)
from src.training.axis_alignment_oof import choose_lambda


class ToyAxisModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.classification_head = nn.Linear(2, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return x


def candidate(value: float, auroc: float, bacc: float, angle: float) -> dict:
    return {
        "lambda_axis": value,
        "auroc": auroc,
        "balanced_accuracy": bacc,
        "mean_head_axis_angle_degrees": angle,
    }


def selection_config() -> dict:
    return {
        "ce_auroc": 0.98,
        "ce_optimized_bacc": 0.94,
        "max_auroc_drop": 0.005,
        "max_bacc_drop": 0.01,
        "numeric_tolerance": 1e-12,
    }


def test_axis_loss_endpoints_and_gradient_scope() -> None:
    model = SourceMedTSTTT(
        dim=8,
        max_channel=2,
        num_heads=4,
        num_layers=1,
        patch_size=8,
        num_classes=2,
    )
    axis = torch.zeros(8)
    axis[0] = 1.0
    with torch.no_grad():
        model.backbone.classification_head.weight.zero_()
        model.backbone.classification_head.weight[1, 0] = 1.0
    assert torch.isclose(axis_alignment_loss(model, axis), torch.tensor(0.0))
    opposite = -axis
    assert torch.isclose(axis_alignment_loss(model, opposite), torch.tensor(2.0))

    rotated = torch.zeros(8)
    rotated[1] = 1.0
    loss = axis_alignment_loss(model, rotated)
    loss.backward()
    head = model.backbone.classification_head
    assert head.weight.grad is not None
    assert head.weight.grad.norm() > 0
    assert head.bias.grad is None
    backbone_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.classification_head")
    ]
    assert all(gradient is None for gradient in backbone_grads)


def test_axis_loss_rejects_attached_axis() -> None:
    model = ToyAxisModel()
    with pytest.raises(ValueError, match="detached"):
        axis_alignment_loss(model, torch.ones(2, requires_grad=True))


def test_source_axis_uses_normalized_class_prototypes() -> None:
    model = ToyAxisModel()
    loader = [
        {"x": torch.tensor([[2.0, 0.0], [0.0, 3.0]]), "y": torch.tensor([0, 1])}
    ]
    axis, stats = extract_source_axis(model, loader, device=torch.device("cpu"))
    expected = torch.tensor([-1.0, 1.0]) / np.sqrt(2.0)
    assert torch.allclose(axis, expected, atol=1e-7)
    assert stats["class_counts"] == {"0": 1, "1": 1}
    assert stats["windows"] == 2


def test_source_axis_rejects_hidden_labels() -> None:
    model = ToyAxisModel()
    loader = [{"x": torch.ones(2, 2), "y": torch.tensor([-1, -1])}]
    with pytest.raises(ValueError, match="visible binary"):
        extract_source_axis(model, loader, device=torch.device("cpu"))


def test_choose_lambda_applies_floors_then_angle_then_smaller_lambda() -> None:
    rows = [
        candidate(0.01, 0.974999999999, 0.94, 10.0),
        candidate(0.05, 0.98, 0.93, 8.0),
        candidate(0.10, 0.98, 0.94, 5.0),
        candidate(0.20, 0.98, 0.94, 5.0),
    ]
    selected, audited = choose_lambda(rows, selection_config())
    assert selected is not None
    assert selected["lambda_axis"] == 0.10
    assert [row["eligible"] for row in audited] == [True, True, True, True]


def test_choose_lambda_freezes_no_eligible_path() -> None:
    rows = [candidate(0.01, 0.90, 0.80, 1.0)]
    selected, audited = choose_lambda(rows, selection_config())
    assert selected is None
    assert audited[0]["eligible"] is False


def test_m2_config_and_lambda_slug() -> None:
    config = json.loads(
        Path("configs/experiments/main_m2_axis_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    validate_m2_config(config)
    assert lambda_slug(0.1) == "lambda_0p10"
    bad = dict(config)
    bad["targets"] = {"forbidden": {}}
    with pytest.raises(ValueError, match="source data only"):
        validate_m2_config(bad)


def test_unit_vector_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError, match="finite nonzero"):
        unit_vector(np.zeros(4), name="test")


def test_binary_head_direction_is_unit_length() -> None:
    model = ToyAxisModel()
    assert torch.isclose(binary_head_direction(model).norm(), torch.tensor(1.0))
