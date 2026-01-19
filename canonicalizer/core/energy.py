"""
Energy/Distance Functions for Feature Comparison

Defines distance metrics for comparing feature representations in the
energy-based canonicalization framework.

The energy function E(g; I_S, I_T) = d(φ(g · I_S), φ(I_T)) measures how
well the warped source features align with the target features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


def cosine_distance(
    features1: torch.Tensor,
    features2: torch.Tensor,
    dim: int = -1,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> torch.Tensor:
    """
    Compute cosine distance between feature tensors.

    Cosine distance = 1 - cosine_similarity

    Args:
        features1: (B, ..., D) first feature tensor
        features2: (B, ..., D) second feature tensor
        dim: Dimension along which to compute cosine similarity
        reduction: How to reduce spatial dimensions
                  'none': return per-element distances
                  'mean': average over all except batch dimension
                  'sum': sum over all except batch dimension

    Returns:
        Distance tensor (shape depends on reduction)
    """
    # Normalize features
    f1_norm = F.normalize(features1, dim=dim, p=2)
    f2_norm = F.normalize(features2, dim=dim, p=2)

    # Cosine similarity
    cos_sim = (f1_norm * f2_norm).sum(dim=dim)

    # Cosine distance
    cos_dist = 1 - cos_sim

    if reduction == 'none':
        return cos_dist
    elif reduction == 'mean':
        # Average over all spatial dimensions, keep batch
        return cos_dist.flatten(1).mean(dim=1)
    elif reduction == 'sum':
        return cos_dist.flatten(1).sum(dim=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def l2_distance(
    features1: torch.Tensor,
    features2: torch.Tensor,
    reduction: Literal['none', 'mean', 'sum'] = 'mean',
    squared: bool = False
) -> torch.Tensor:
    """
    Compute L2 (Euclidean) distance between feature tensors.

    Args:
        features1: (B, ..., D) first feature tensor
        features2: (B, ..., D) second feature tensor
        reduction: How to reduce spatial dimensions
        squared: If True, return squared L2 distance

    Returns:
        Distance tensor (shape depends on reduction)
    """
    diff = features1 - features2
    sq_dist = (diff ** 2).sum(dim=-1)

    if not squared:
        dist = torch.sqrt(sq_dist + 1e-8)
    else:
        dist = sq_dist

    if reduction == 'none':
        return dist
    elif reduction == 'mean':
        return dist.flatten(1).mean(dim=1)
    elif reduction == 'sum':
        return dist.flatten(1).sum(dim=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def smooth_l1_distance(
    features1: torch.Tensor,
    features2: torch.Tensor,
    beta: float = 1.0,
    reduction: Literal['none', 'mean', 'sum'] = 'mean'
) -> torch.Tensor:
    """
    Compute smooth L1 (Huber) distance between feature tensors.

    More robust to outliers than L2.

    Args:
        features1: (B, ..., D) first feature tensor
        features2: (B, ..., D) second feature tensor
        beta: Threshold for switching between L1 and L2
        reduction: How to reduce spatial dimensions

    Returns:
        Distance tensor
    """
    dist = F.smooth_l1_loss(features1, features2, reduction='none', beta=beta)
    dist = dist.sum(dim=-1)  # Sum over feature dimension

    if reduction == 'none':
        return dist
    elif reduction == 'mean':
        return dist.flatten(1).mean(dim=1)
    elif reduction == 'sum':
        return dist.flatten(1).sum(dim=1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class EnergyFunction(nn.Module):
    """
    Energy function for canonicalization.

    Computes E(g; I_S, I_T) = d(φ(g · I_S), φ(I_T))

    This module only computes the distance given pre-computed features.
    The feature extraction and warping are handled externally.
    """

    def __init__(
        self,
        distance_type: Literal['cosine', 'l2', 'smooth_l1'] = 'cosine',
        reduction: Literal['mean', 'sum'] = 'mean',
        **kwargs
    ):
        """
        Args:
            distance_type: Type of distance metric
            reduction: How to aggregate spatial distances
            **kwargs: Additional arguments for specific distance functions
        """
        super().__init__()
        self.distance_type = distance_type
        self.reduction = reduction
        self.kwargs = kwargs

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute energy (distance) between feature tensors.

        Args:
            source_features: (B, N, D) or (B, H, W, D) warped source features
            target_features: (B, N, D) or (B, H, W, D) target features
            mask: Optional (B, N) or (B, H, W) mask for valid regions

        Returns:
            (B,) energy values
        """
        if self.distance_type == 'cosine':
            dist = cosine_distance(
                source_features, target_features,
                dim=-1, reduction='none'
            )
        elif self.distance_type == 'l2':
            dist = l2_distance(
                source_features, target_features,
                reduction='none', squared=self.kwargs.get('squared', False)
            )
        elif self.distance_type == 'smooth_l1':
            dist = smooth_l1_distance(
                source_features, target_features,
                beta=self.kwargs.get('beta', 1.0),
                reduction='none'
            )
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        # Apply mask if provided
        if mask is not None:
            dist = dist * mask.float()
            if self.reduction == 'mean':
                return dist.flatten(1).sum(dim=1) / (mask.flatten(1).sum(dim=1) + 1e-8)
            else:  # sum
                return dist.flatten(1).sum(dim=1)
        else:
            if self.reduction == 'mean':
                return dist.flatten(1).mean(dim=1)
            else:
                return dist.flatten(1).sum(dim=1)


class PatchWiseEnergy(nn.Module):
    """
    Patch-wise energy using optimal transport or soft assignment.

    Instead of comparing features element-wise, this finds the best
    matching between source and target patches.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3
    ):
        """
        Args:
            temperature: Temperature for softmax in matching
            sinkhorn_iterations: Number of Sinkhorn iterations for OT
        """
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute patch-wise energy using soft assignment.

        Args:
            source_features: (B, N, D) source patch features
            target_features: (B, M, D) target patch features

        Returns:
            (B,) energy values
        """
        B, N, D = source_features.shape
        M = target_features.shape[1]

        # Normalize features
        src_norm = F.normalize(source_features, dim=-1)
        tgt_norm = F.normalize(target_features, dim=-1)

        # Compute similarity matrix: (B, N, M)
        similarity = torch.bmm(src_norm, tgt_norm.transpose(1, 2))

        # Apply Sinkhorn normalization for doubly-stochastic assignment
        log_alpha = similarity / self.temperature

        for _ in range(self.sinkhorn_iterations):
            # Row normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            # Column normalization
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        # Soft assignment matrix
        assignment = torch.exp(log_alpha)

        # Energy is negative of matched similarity
        matched_sim = (assignment * similarity).sum(dim=(1, 2))
        energy = -matched_sim / N  # Normalize by number of source patches

        return energy


def test_energy():
    """Test energy functions."""
    import torch

    print("Testing energy functions...")

    B, N, D = 2, 100, 256

    # Test 1: Cosine distance of identical features should be 0
    features = torch.randn(B, N, D)
    dist = cosine_distance(features, features)
    assert torch.allclose(dist, torch.zeros(B), atol=1e-6), "Same features should have 0 cosine distance"
    print("✓ Cosine distance of identical features is 0")

    # Test 2: Cosine distance of orthogonal features should be 1
    f1 = torch.zeros(1, 1, 2)
    f1[0, 0, 0] = 1
    f2 = torch.zeros(1, 1, 2)
    f2[0, 0, 1] = 1
    dist = cosine_distance(f1, f2)
    assert torch.allclose(dist, torch.ones(1), atol=1e-6), "Orthogonal features should have distance 1"
    print("✓ Cosine distance of orthogonal features is 1")

    # Test 3: L2 distance
    f1 = torch.tensor([[[0.0, 0.0]]])
    f2 = torch.tensor([[[3.0, 4.0]]])
    dist = l2_distance(f1, f2)
    assert torch.allclose(dist, torch.tensor([5.0]), atol=1e-6), "L2 distance should be 5"
    print("✓ L2 distance correct")

    # Test 4: EnergyFunction module
    energy_fn = EnergyFunction(distance_type='cosine')
    features = torch.randn(B, N, D)
    energy = energy_fn(features, features)
    assert torch.allclose(energy, torch.zeros(B), atol=1e-6), "Energy of same features should be 0"
    print("✓ EnergyFunction module works")

    # Test 5: EnergyFunction with mask
    energy_fn = EnergyFunction(distance_type='cosine', reduction='mean')
    f1 = torch.randn(2, 10, 64)
    f2 = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[0, 5:] = 0  # Mask out second half for first batch

    energy_masked = energy_fn(f1, f2, mask=mask)
    assert energy_masked.shape == (2,), "Energy should have batch dimension"
    print("✓ EnergyFunction with mask works")

    # Test 6: Gradient flow
    f1 = torch.randn(B, N, D, requires_grad=True)
    f2 = torch.randn(B, N, D)
    energy = cosine_distance(f1, f2)
    energy.sum().backward()
    assert f1.grad is not None, "Gradients should flow through cosine distance"
    print("✓ Gradients flow through energy functions")

    # Test 7: PatchWiseEnergy
    patch_energy = PatchWiseEnergy(temperature=0.1)
    f1 = torch.randn(B, 49, D)  # 7x7 patches
    f2 = torch.randn(B, 49, D)
    energy = patch_energy(f1, f2)
    assert energy.shape == (B,), "Patch energy should have batch dimension"
    print("✓ PatchWiseEnergy works")

    print("\nAll energy tests passed!")


if __name__ == "__main__":
    test_energy()
