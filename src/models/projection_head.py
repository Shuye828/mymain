"""Projection heads for later representation-learning experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Map backbone embeddings to unit-normalized projection features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        *,
        kind: str = "mlp",
        hidden_dim: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        if eps <= 0:
            raise ValueError("projection normalization epsilon must be positive")
        if kind == "linear":
            self.projector: nn.Module = nn.Linear(input_dim, output_dim)
        elif kind == "mlp":
            resolved_hidden = input_dim if hidden_dim is None else hidden_dim
            if resolved_hidden <= 0:
                raise ValueError("projection hidden dimension must be positive")
            self.projector = nn.Sequential(
                nn.Linear(input_dim, resolved_hidden),
                nn.GELU(),
                nn.Linear(resolved_hidden, output_dim),
            )
        else:
            raise ValueError(f"unsupported projection kind {kind!r}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kind = kind
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [B,{self.input_dim}] features, got "
                f"{tuple(features.shape)}"
            )
        projected = self.projector(features)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("projection head produced non-finite values")
        return F.normalize(projected, dim=-1, eps=self.eps)
