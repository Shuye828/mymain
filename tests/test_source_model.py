import torch

from src.models.medts_ttt_wrapper import SourceMedTSTTT


def test_source_model_returns_binary_logits() -> None:
    model = SourceMedTSTTT(
        dim=16,
        max_channel=2,
        num_heads=4,
        num_layers=1,
        patch_size=8,
        num_classes=2,
    )
    output = model(torch.randn(2, 2, 256))

    assert output.shape == (2, 2)
