
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.warping import transform_points

class PhotometricLoss(nn.Module):
    """
    Photometric loss for dense matching.
    Measures similarity between warped source and target in feature space.

    This is the primary loss for training with dense matchers where we
    don't have sparse keypoint correspondences.
    """
    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(
        self,
        warped_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            warped_features: (B, N, D) features of warped source
            target_features: (B, N, D) features of target
            mask: (B, N) optional valid region mask

        Returns:
            Scalar loss
        """
        if self.loss_type == 'l1':
            loss = F.l1_loss(warped_features, target_features, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(warped_features, target_features, reduction='none')
        elif self.loss_type == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            loss = 1 - F.cosine_similarity(warped_features, target_features, dim=-1, eps=1e-8)
            if len(loss.shape) == 2:  # (B, N)
                loss = loss.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Average over feature dimension if needed
        if len(loss.shape) == 3:  # (B, N, D)
            loss = loss.mean(dim=-1)  # (B, N)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()

        return loss

class CorrespondenceLoss(nn.Module):
    """
    Supervises the predicted transformation using ground truth correspondences.
    
    If g* aligns Source to Target, then for a correspondence pair (p_s, p_t),
    the transformed point g* · p_s should be close to p_t.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self, 
        g_star: torch.Tensor, 
        source_points: torch.Tensor, 
        target_points: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            g_star: (B, 3, 3) predicted transformation
            source_points: (B, N, 2) points in source image
            target_points: (B, N, 2) corresponding points in target image
            weights: (B, N) optional weights for each correspondence
            
        Returns:
            Scalar loss
        """
        # Transform source points by g_star
        transformed_points = transform_points(source_points, g_star)
        
        # Squared Euclidean distance
        dist_sq = ((transformed_points - target_points) ** 2).sum(dim=-1) # (B, N)
        
        if weights is not None:
            dist_sq = dist_sq * weights
            
        if self.reduction == 'mean':
            if weights is not None:
                return dist_sq.sum() / (weights.sum() + 1e-8)
            return dist_sq.mean()
        elif self.reduction == 'sum':
            return dist_sq.sum()
        else:
            return dist_sq

class RegularizationLoss(nn.Module):
    """
    Regularizes the transformation parameters to avoid extreme distortions.
    """
    def __init__(self, weight_omega=0.0, weight_sigma=0.0, weight_t=0.0):
        super().__init__()
        self.weights = [weight_omega, weight_sigma, weight_t]
        
    def forward(self, g_star: torch.Tensor) -> torch.Tensor:
        from ..core.lie_group import Sim2
        
        # log map to get Lie algebra parameters
        xi = Sim2.log(g_star) # (B, 4) -> (omega, sigma, vx, vy)
        
        omega = xi[:, 0]
        sigma = xi[:, 1]
        v = xi[:, 2:]
        
        loss = 0.0
        if self.weights[0] > 0:
            loss += self.weights[0] * (omega ** 2).mean()
        if self.weights[1] > 0:
            loss += self.weights[1] * (sigma ** 2).mean()
        if self.weights[2] > 0:
            loss += self.weights[2] * (v ** 2).sum(dim=-1).mean()
            
        return loss
