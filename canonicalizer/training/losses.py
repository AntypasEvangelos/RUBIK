
import torch
import torch.nn as nn
from ..core.warping import transform_points

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
