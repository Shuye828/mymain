import pytest
import torch

from src.adaptation.disease_direction import (
    SourcePrototypeAccumulator,
    estimate_disease_direction,
)


def test_disease_direction_points_from_nonaf_to_af() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    result = estimate_disease_direction(features, labels)
    scores = features @ result.direction

    assert result.nonaf_count == 2
    assert result.af_count == 2
    assert torch.allclose(
        torch.linalg.vector_norm(result.direction), torch.tensor(1.0)
    )
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_streaming_prototypes_match_single_batch_result() -> None:
    torch.manual_seed(4)
    features = torch.randn(12, 5)
    labels = torch.tensor([0, 1] * 6)
    direct = estimate_disease_direction(features, labels)
    accumulator = SourcePrototypeAccumulator(5)
    accumulator.update(features[:5], labels[:5])
    accumulator.update(features[5:], labels[5:])
    streamed = accumulator.finalize()

    assert torch.allclose(direct.nonaf_prototype, streamed.nonaf_prototype)
    assert torch.allclose(direct.af_prototype, streamed.af_prototype)
    assert torch.allclose(direct.direction, streamed.direction)


def test_source_prototypes_reject_hidden_target_labels() -> None:
    accumulator = SourcePrototypeAccumulator(3)

    with pytest.raises(ValueError, match="visible binary source labels"):
        accumulator.update(torch.randn(2, 3), torch.tensor([-1, -1]))


def test_source_prototypes_require_both_classes() -> None:
    accumulator = SourcePrototypeAccumulator(3)
    accumulator.update(torch.randn(2, 3), torch.tensor([0, 0]))

    with pytest.raises(ValueError, match="both source classes"):
        accumulator.finalize()


def test_source_accumulator_converts_after_cpu_transfer(monkeypatch) -> None:
    features = torch.randn(2, 3)
    labels = torch.tensor([0, 1])
    original_to = torch.Tensor.to

    def guarded_to(self, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("dtype") == torch.float64:
            raise AssertionError("combined device and float64 conversion")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", guarded_to)
    accumulator = SourcePrototypeAccumulator(3)
    accumulator.update(features, labels)

    result = accumulator.finalize()
    assert result.nonaf_count == 1
    assert result.af_count == 1
