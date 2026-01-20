
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

from ..core.lie_group import Sim2
from ..core.warping import warp_image
from ..core.energy import EnergyFunction
from .feature_extractor import DINOv2FeatureExtractor
from .cross_attention import CrossAttentionBlock

class FrozenEnergyCanonicalizer(nn.Module):
    """
    Canonicalizer using frozen DINOv2 energy landscape.

    Approach 1:
        1. Extract frozen features.
        2. Predict rough initialization using cross-attention.
        3. Refine transformation via gradient descent on the frozen energy.

    COORDINATE SYSTEM:
        - The transformation g_star operates in NORMALIZED coordinates [-1, 1]
        - warp_image() internally handles conversion from normalized to pixel coords
        - This is consistent with grid_sample's coordinate convention
        - When using keypoints, they must be in NORMALIZED coordinates as well
    """
    def __init__(
        self,
        dino_model: str = 'dinov2_vitb14',
        num_iterations: int = 5,
        step_size: float = 0.5,
        distance_type: str = 'cosine',
        feature_layer: str = 'x_norm_patchtokens'
    ):
        super().__init__()
        
        # 1. Feature Extractor (Frozen)
        self.dino = DINOv2FeatureExtractor(model_name=dino_model, freeze=True)
        self.feature_layer = feature_layer

        # Determine embedding dimension
        if 'vits' in dino_model:
            embed_dim = 384
        elif 'vitb' in dino_model:
            embed_dim = 768
        elif 'vitl' in dino_model:
            embed_dim = 1024
        elif 'vitg' in dino_model:
            embed_dim = 1536
        else:
            embed_dim = 768 # Default fallback
        
        # 2. Initialization Network (Learned)
        self.cross_attention = CrossAttentionBlock(dim=embed_dim, num_heads=8)
        self.init_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # (omega, sigma, vx, vy)
        )
        # Initialize small for stability
        nn.init.constant_(self.init_head[-1].weight, 0)
        nn.init.constant_(self.init_head[-1].bias, 0)
        
        # 3. Optimization hyperparameters
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.energy_fn = EnergyFunction(distance_type=distance_type, reduction='mean')

    def forward(
        self, 
        source_image: torch.Tensor, 
        target_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_image: (B, C, H, W)
            target_image: (B, C, H, W)
            
        Returns:
            warped_source: (B, C, H, W)
            g_star: (B, 3, 3) Estimated Sim(2) transformation
        """
        B = source_image.shape[0]
        
        # 1. Extract Target Features
        with torch.no_grad():
            feats_t_dict = self.dino(target_image)
            feats_t = feats_t_dict[self.feature_layer]  # (B, N, 768)
        
        # 2. Initialization
        # Extract source features once for initialization
        with torch.no_grad():
            feats_s_init_dict = self.dino(source_image)
            feats_s_init = feats_s_init_dict[self.feature_layer]
            
        # Condition on target
        feats_cond = self.cross_attention(feats_s_init, feats_t)
        
        # Predict params
        xi_0 = self.init_head(feats_cond.mean(dim=1)) # Global pool
        g_current = Sim2.exp(xi_0)
        
        # 3. Iterative Refinement
        # We perform RIGHT-INVARIANT GRADIENT DESCENT on the Sim(2) manifold:
        #   - Compute energy gradient w.r.t. perturbation at identity
        #   - Update via right composition: g_{k+1} = g_k · exp(-α·∇ξE)
        #
        # This is a standard Lie group optimization technique that respects
        # the group structure and avoids coordinate singularities.
        #
        # GRADIENT FLOW:
        #   - DINO features are frozen (no grad w.r.t. DINO params)
        #   - We backprop through: warp → features → energy
        #   - Gradient w.r.t. xi_perturb is computed, then used to update g_current
        #   - Using FOMAML approximation (detach grad) to avoid expensive 2nd order backprop

        for i in range(self.num_iterations):
            # Prepare perturbation variable at Identity in Lie algebra
            # This represents a small change Δξ ∈ sim(2) at the current point
            xi_perturb = torch.zeros(B, 4, device=source_image.device, requires_grad=True)
            g_perturb = Sim2.exp(xi_perturb)

            # Right composition: Apply perturbation to current transformation
            # g_candidate = g_current · g_perturb
            g_candidate = Sim2.compose(g_current, g_perturb)
            
            # Warp
            warped_candidate = warp_image(source_image, g_candidate)
            
            # Extract features of warped image
            # Note: We can't use torch.no_grad() here because we need grad wrt warped_candidate
            # to flow back to xi_perturb. DINO parameters don't require grad, but input does.
            feats_warped_dict = self.dino(warped_candidate)
            feats_warped = feats_warped_dict[self.feature_layer]
            
            # Compute Energy
            energy = self.energy_fn(feats_warped, feats_t)
            
            # Compute Gradient wrt perturbation
            # create_graph=True allows backprop from the final loss through this gradient step
            grad_xi = torch.autograd.grad(energy.sum(), xi_perturb, create_graph=self.training)[0]
            
            # Detach gradient to avoid double backward (FOMAML approximation)
            # This avoids "derivative for aten::grid_sampler_2d_backward is not implemented"
            grad_xi = grad_xi.detach()
            
            # Update
            # Gradient checks: if energy increases with xi, we want -grad
            delta_xi = -self.step_size * grad_xi
            
            # Apply update
            g_update = Sim2.exp(delta_xi)
            g_current = Sim2.compose(g_current, g_update)
            
        # Final warp
        warped_source = warp_image(source_image, g_current)
        
        return warped_source, g_current
