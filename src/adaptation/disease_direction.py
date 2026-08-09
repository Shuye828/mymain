"""Source-only class prototypes and AF disease-direction estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DiseaseDirection:
    """Two source class prototypes and the normalized non-AF-to-AF vector."""

    nonaf_prototype: torch.Tensor
    af_prototype: torch.Tensor
    direction: torch.Tensor
    nonaf_count: int
    af_count: int


class SourcePrototypeAccumulator:
    """Accumulate source-only class means without storing every feature."""

    def __init__(self, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature dimension must be positive")
        self.feature_dim = feature_dim
        self._sums = torch.zeros(2, feature_dim, dtype=torch.float64)
        self._counts = torch.zeros(2, dtype=torch.long)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected [B,{self.feature_dim}] features, got "
                f"{tuple(features.shape)}"
            )
        labels = labels.reshape(-1)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("feature and label batch sizes differ")
        if not torch.isfinite(features).all():
            raise FloatingPointError("source features contain NaN or Inf")
        unique = set(int(value) for value in labels.detach().cpu().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                "source prototypes require visible binary source labels 0/1"
            )
        # MPS cannot perform a direct device-and-float64 conversion. Move the
        # tensor first, then accumulate in CPU float64 for stable class means.
        cpu_features = features.detach().cpu().to(dtype=torch.float64)
        cpu_labels = labels.detach().to(device="cpu", dtype=torch.long)
        for label in (0, 1):
            mask = cpu_labels == label
            count = int(mask.sum().item())
            if count:
                self._sums[label] += cpu_features[mask].sum(dim=0)
                self._counts[label] += count

    def finalize(self, *, eps: float = 1e-12) -> DiseaseDirection:
        if eps <= 0:
            raise ValueError("normalization epsilon must be positive")
        if bool((self._counts == 0).any()):
            raise ValueError("both source classes are required for prototypes")
        prototypes = self._sums / self._counts[:, None]
        difference = prototypes[1] - prototypes[0]
        norm = torch.linalg.vector_norm(difference)
        if not torch.isfinite(norm) or float(norm) <= eps:
            raise ValueError("source class prototypes define no finite direction")
        direction = F.normalize(difference, dim=0, eps=eps)
        return DiseaseDirection(
            nonaf_prototype=prototypes[0].to(torch.float32),
            af_prototype=prototypes[1].to(torch.float32),
            direction=direction.to(torch.float32),
            nonaf_count=int(self._counts[0]),
            af_count=int(self._counts[1]),
        )


def estimate_disease_direction(
    features: torch.Tensor, labels: torch.Tensor
) -> DiseaseDirection:
    """Estimate the non-AF-to-AF direction from visible source labels only."""

    if features.ndim != 2:
        raise ValueError(f"expected [N,D] features, got {tuple(features.shape)}")
    accumulator = SourcePrototypeAccumulator(features.shape[1])
    accumulator.update(features, labels)
    return accumulator.finalize()
