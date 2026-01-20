
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math

from ..core.lie_group import Sim2
from ..core.warping import warp_image
from .feature_extractor import DINOv2FeatureExtractor
from .cross_attention import CrossAttentionBlock


class LocalUpdateNetwork(nn.Module):
    """
    Update network for local (patch-based) DEQ canonicalizer.

    Predicts spatially-varying Sim(2) transformations where
    different image patches can have different transformation parameters.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        patch_size: int = 64,
        output_resolution: int = 8  # Number of patches per dimension
    ):
        super().__init__()

        self.patch_size = patch_size
        self.output_resolution = output_resolution

        # Cross-attention for patch-wise conditioning
        self.cross_attention = CrossAttentionBlock(dim=embed_dim, num_heads=8)

        # Convolutional layers to process patch features spatially
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Predict Sim(2) parameters per spatial location
        # Output: 4 channels (omega, sigma, vx, vy) at each spatial location
        self.param_head = nn.Conv2d(hidden_dim, 4, kernel_size=1)

        # Initialize to output small updates
        nn.init.constant_(self.param_head.weight, 0)
        nn.init.constant_(self.param_head.bias, 0)

    def forward(
        self,
        source_feats: torch.Tensor,
        target_feats: torch.Tensor,
        H_img: int,
        W_img: int
    ) -> torch.Tensor:
        """
        Args:
            source_feats: (B, N, D) patch features from warped source
            target_feats: (B, N, D) patch features from target
            H_img: Original image height
            W_img: Original image width

        Returns:
            delta_xi_field: (B, 4, H_grid, W_grid) spatially-varying Lie algebra updates
        """
        B, N, D = source_feats.shape

        # Determine spatial dimensions of feature map
        # DINOv2 downsamples by 14
        H_feat = H_img // 14
        W_feat = W_img // 14

        # Cross-attention conditioning
        feats_cond = self.cross_attention(source_feats, target_feats)  # (B, N, D)

        # Reshape to spatial grid
        # Need to handle non-square images
        # Assume square for simplicity (can extend to rectangular)
        # For DINOv2 with 518x518 input: H_feat = W_feat = 37
        feat_grid = feats_cond.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)  # (B, D, H', W')

        # Process spatially
        feat_processed = self.conv_layers(feat_grid)  # (B, hidden_dim, H', W')

        # Predict Sim(2) parameters per location
        delta_xi_field = self.param_head(feat_processed)  # (B, 4, H', W')

        # Optionally resize to desired output resolution
        if self.output_resolution != delta_xi_field.shape[2]:
            delta_xi_field = F.interpolate(
                delta_xi_field,
                size=(self.output_resolution, self.output_resolution),
                mode='bilinear',
                align_corners=True
            )

        return delta_xi_field


def create_local_warp_field(
    g_field: torch.Tensor,
    H: int,
    W: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create dense warp field from spatially-varying Sim(2) transformations.

    Args:
        g_field: (B, 3, 3, H_grid, W_grid) Sim(2) transformations at each grid location
        H: Target image height
        W: Target image width
        device: torch device

    Returns:
        grid: (B, H, W, 2) dense warp field in normalized coordinates
    """
    B, _, _, H_grid, W_grid = g_field.shape

    # Create normalized coordinate grid for full image
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (H, W, 3)
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # For each pixel, determine which grid cell it belongs to
    # Map pixel coordinates to grid indices
    # Normalized coords [-1, 1] -> grid index [0, H_grid-1]
    grid_y = ((coords[..., 1] + 1) / 2) * (H_grid - 1)  # (B, H, W)
    grid_x = ((coords[..., 0] + 1) / 2) * (W_grid - 1)  # (B, H, W)

    # Bilinear interpolation of transformation parameters
    # For simplicity, use nearest neighbor here
    # (Can extend to bilinear interpolation of Lie algebra parameters)
    grid_y_idx = torch.clamp(grid_y.long(), 0, H_grid - 1)
    grid_x_idx = torch.clamp(grid_x.long(), 0, W_grid - 1)

    # Gather transformations
    # g_field: (B, 3, 3, H_grid, W_grid)
    # We need to gather for each pixel
    warped_coords = torch.zeros(B, H, W, 2, device=device)

    for b in range(B):
        for i in range(H):
            for j in range(W):
                gy = grid_y_idx[b, i, j]
                gx = grid_x_idx[b, i, j]
                g_local = g_field[b, :, :, gy, gx]  # (3, 3)

                # Apply transformation to this pixel
                coord_h = coords[b, i, j]  # (3,)
                coord_warped_h = g_local @ coord_h  # (3,)
                coord_warped = coord_warped_h[:2] / coord_warped_h[2]  # (2,)

                warped_coords[b, i, j] = coord_warped

    return warped_coords


def warp_image_local(
    image: torch.Tensor,
    g_field: torch.Tensor,
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Warp image using spatially-varying transformations.

    Args:
        image: (B, C, H, W)
        g_field: (B, 3, 3, H_grid, W_grid) transformation field

    Returns:
        warped: (B, C, H, W)
    """
    B, C, H, W = image.shape
    device = image.device

    # Create dense warp field
    grid = create_local_warp_field(g_field, H, W, device)  # (B, H, W, 2)

    # Apply warping
    warped = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    )

    return warped


class LocalDEQCanonicalizer(nn.Module):
    """
    Local (patch-based) DEQ canonicalizer.

    Instead of a single global Sim(2) transformation, this model predicts
    spatially-varying transformations where different image regions can
    undergo different geometric transformations.

    This is useful for:
    - Non-planar scenes
    - Objects at different depths
    - Handling local deformations

    COORDINATE SYSTEM:
        - Each patch has its own Sim(2) transformation in normalized coords [-1, 1]
        - Transformations are smoothly interpolated across the image
    """
    def __init__(
        self,
        dino_model: str = 'dinov2_vitb14',
        freeze_dino: bool = True,
        patch_resolution: int = 8,  # Grid resolution for transformations
        max_iterations: int = 20,
        convergence_threshold: float = 1e-3
    ):
        super().__init__()

        # Feature extractor
        self.dino = DINOv2FeatureExtractor(model_name=dino_model, freeze=freeze_dino)

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
            embed_dim = 768

        # Local update network
        self.update_network = LocalUpdateNetwork(
            embed_dim=embed_dim,
            output_resolution=patch_resolution
        )

        self.feature_layer = 'x_norm_patchtokens'
        self.patch_resolution = patch_resolution
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the spatially-varying transformation via DEQ.

        Args:
            source_image: (B, C, H, W)
            target_image: (B, C, H, W)
            return_trajectory: If True, return iteration trajectory

        Returns:
            warped_source: (B, C, H, W)
            g_field: (B, 3, 3, H_grid, W_grid) transformation field
            trajectory (optional): List of update norms
        """
        B, C, H, W = source_image.shape
        device = source_image.device

        # Extract target features
        with torch.no_grad() if self.dino.model_name else torch.enable_grad():
            target_feats = self.dino(target_image)[self.feature_layer]

        # Initialize transformation field at identity
        g_field = torch.eye(3, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        g_field = g_field.expand(B, 3, 3, self.patch_resolution, self.patch_resolution)

        trajectory = []

        # Fixed point iteration
        for iteration in range(self.max_iterations):
            # Warp source by current transformation field
            warped_source = warp_image_local(source_image, g_field)

            # Extract features
            source_feats = self.dino(warped_source)[self.feature_layer]

            # Predict local updates
            delta_xi_field = self.update_network(
                source_feats, target_feats, H, W
            )  # (B, 4, H_grid, W_grid)

            # Check convergence
            update_norm = torch.norm(delta_xi_field.reshape(B, -1), dim=-1).mean()

            if return_trajectory:
                trajectory.append(update_norm.item())

            if update_norm < self.convergence_threshold:
                break

            # Apply updates to each grid location
            B, _, H_grid, W_grid = delta_xi_field.shape
            g_field_new = torch.zeros_like(g_field)

            for b in range(B):
                for i in range(H_grid):
                    for j in range(W_grid):
                        delta_xi = delta_xi_field[b, :, i, j]  # (4,)
                        g_update = Sim2.exp(delta_xi.unsqueeze(0)).squeeze(0)  # (3, 3)
                        g_current = g_field[b, :, :, i, j]  # (3, 3)
                        g_field_new[b, :, :, i, j] = g_current @ g_update

            g_field = g_field_new

        # Final warp
        warped_source = warp_image_local(source_image, g_field)

        if return_trajectory:
            return warped_source, g_field, trajectory
        else:
            return warped_source, g_field

    def get_global_transformation(self, g_field: torch.Tensor) -> torch.Tensor:
        """
        Extract a single global transformation by averaging the field.

        Args:
            g_field: (B, 3, 3, H_grid, W_grid)

        Returns:
            g_global: (B, 3, 3)
        """
        B, _, _, H_grid, W_grid = g_field.shape

        # Average Lie algebra parameters (more principled than averaging matrices)
        xi_sum = torch.zeros(B, 4, device=g_field.device)

        for b in range(B):
            for i in range(H_grid):
                for j in range(W_grid):
                    g_local = g_field[b, :, :, i, j].unsqueeze(0)  # (1, 3, 3)
                    xi_local = Sim2.log(g_local).squeeze(0)  # (4,)
                    xi_sum[b] += xi_local

        xi_avg = xi_sum / (H_grid * W_grid)
        g_global = Sim2.exp(xi_avg)

        return g_global
