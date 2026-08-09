"""Stage-3 source classifier wrapper without representation changes."""

from __future__ import annotations

import torch
from torch import nn

from MedTS_TTT import MedTSTTT


class SourceMedTSTTT(nn.Module):
    """Preserve upstream logits-only behavior for CE source training."""

    def __init__(
        self,
        *,
        dim: int = 128,
        max_channel: int = 2,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = 8,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("stage-3 source classifier requires two classes")
        self.backbone = MedTSTTT(
            dim=dim,
            max_channel=max_channel,
            num_heads=num_heads,
            num_layers=num_layers,
            patch_size=patch_size,
            num_classes=num_classes,
        )

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(f"expected [B,C,T], got {tuple(x.shape)}")
        if x.shape[1] != 2:
            raise ValueError(f"expected two ECG channels, got {x.shape[1]}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unprojected mean-pooled MedTS-TTT embedding."""

        self._validate_input(x)
        features = self.backbone.forward_features(x)
        expected = (x.shape[0], self.backbone.classification_head.in_features)
        if features.shape != expected:
            raise ValueError(f"unexpected feature shape {tuple(features.shape)}")
        return features

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(x)
        if return_features:
            logits, features = self.backbone(x, return_features=True)
        else:
            logits = self.backbone(x)
        if logits.shape != (x.shape[0], 2):
            raise ValueError(f"unexpected logits shape {tuple(logits.shape)}")
        if return_features:
            expected = (
                x.shape[0],
                self.backbone.classification_head.in_features,
            )
            if features.shape != expected:
                raise ValueError(
                    f"unexpected feature shape {tuple(features.shape)}"
                )
            return logits, features
        return logits
