import torch

from src.models.medts_ttt_wrapper import SourceMedTSTTT
from src.models.projection_head import ProjectionHead


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


def test_feature_interface_preserves_logits_and_checkpoint_keys() -> None:
    torch.manual_seed(7)
    model = SourceMedTSTTT(
        dim=16,
        max_channel=2,
        num_heads=4,
        num_layers=1,
        patch_size=8,
        num_classes=2,
    ).eval()
    x = torch.randn(2, 2, 256)
    original_keys = set(model.state_dict())

    with torch.inference_mode():
        logits_only = model(x)
        logits_with_features, features = model(x, return_features=True)
        features_only = model.forward_features(x)
        logits_from_features = model.backbone.classification_head(features)

    assert torch.equal(logits_only, logits_with_features)
    assert torch.equal(features, features_only)
    assert torch.equal(logits_only, logits_from_features)
    assert features.shape == (2, 16)
    clone = SourceMedTSTTT(
        dim=16,
        max_channel=2,
        num_heads=4,
        num_layers=1,
        patch_size=8,
        num_classes=2,
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    assert set(clone.state_dict()) == original_keys


def test_projection_head_outputs_unit_norm_features() -> None:
    head = ProjectionHead(16, 8, kind="mlp")
    projected = head(torch.randn(5, 16))

    assert projected.shape == (5, 8)
    assert torch.allclose(
        torch.linalg.vector_norm(projected, dim=1),
        torch.ones(5),
        atol=1e-6,
    )
