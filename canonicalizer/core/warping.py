"""
Differentiable Image Warping

Implements inverse warping for Sim(2) transformations using PyTorch's
grid_sample for differentiability.

For a transformation g ∈ Sim(2), the group action on image I is:
    (g · I)(p) = I(g⁻¹ · p)

This is inverse warping: for each destination pixel p, we ask "where did
this come from?" and sample from the source image at that location.

Inverse warping guarantees every destination pixel gets exactly one value
(no holes), which is essential for neural network training.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .lie_group import Sim2


def create_normalized_grid(
    height: int,
    width: int,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Create a normalized coordinate grid for grid_sample.

    PyTorch's grid_sample expects coordinates in [-1, 1] where:
        (-1, -1) = top-left corner
        (1, 1) = bottom-right corner

    Args:
        height: Image height
        width: Image width
        device: Torch device
        dtype: Torch dtype

    Returns:
        (H, W, 2) tensor of (x, y) coordinates in [-1, 1]
    """
    # Create coordinate ranges
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)

    # Create meshgrid (indexing='xy' gives x varying along columns)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    # Stack to get (H, W, 2) with (x, y) at each position
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return grid


def create_pixel_grid(
    height: int,
    width: int,
    device: torch.device = None,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Create a pixel coordinate grid.

    Args:
        height: Image height
        width: Image width
        device: Torch device
        dtype: Torch dtype

    Returns:
        (H, W, 2) tensor of (x, y) pixel coordinates
        x ranges from 0 to W-1, y ranges from 0 to H-1
    """
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return grid


def pixel_to_normalized(
    coords: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Convert pixel coordinates to normalized [-1, 1] coordinates.

    Args:
        coords: (..., 2) tensor of (x, y) pixel coordinates
        height: Image height
        width: Image width

    Returns:
        (..., 2) tensor of normalized coordinates
    """
    # Normalize: pixel (0, 0) -> (-1, -1), pixel (W-1, H-1) -> (1, 1)
    normalized = coords.clone()
    normalized[..., 0] = 2 * coords[..., 0] / (width - 1) - 1
    normalized[..., 1] = 2 * coords[..., 1] / (height - 1) - 1
    return normalized


def normalized_to_pixel(
    coords: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Convert normalized [-1, 1] coordinates to pixel coordinates.

    Args:
        coords: (..., 2) tensor of normalized coordinates
        height: Image height
        width: Image width

    Returns:
        (..., 2) tensor of pixel coordinates
    """
    pixel = coords.clone()
    pixel[..., 0] = (coords[..., 0] + 1) * (width - 1) / 2
    pixel[..., 1] = (coords[..., 1] + 1) * (height - 1) / 2
    return pixel


def to_homogeneous(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D coordinates to homogeneous coordinates.

    Args:
        coords: (..., 2) tensor of (x, y) coordinates

    Returns:
        (..., 3) tensor of (x, y, 1) homogeneous coordinates
    """
    ones = torch.ones_like(coords[..., :1])
    return torch.cat([coords, ones], dim=-1)


def from_homogeneous(coords: torch.Tensor) -> torch.Tensor:
    """
    Convert homogeneous coordinates to 2D coordinates.

    Args:
        coords: (..., 3) tensor of (x, y, w) homogeneous coordinates

    Returns:
        (..., 2) tensor of (x/w, y/w) coordinates
    """
    return coords[..., :2] / (coords[..., 2:3] + 1e-8)


def apply_transform_to_grid(
    g: torch.Tensor,
    grid: torch.Tensor
) -> torch.Tensor:
    """
    Apply Sim(2) transformation to a coordinate grid.

    Args:
        g: (B, 3, 3) transformation matrices
        grid: (H, W, 2) coordinate grid (same for all batch elements)

    Returns:
        (B, H, W, 2) transformed coordinates
    """
    B = g.shape[0]
    H, W = grid.shape[:2]
    device = g.device
    dtype = g.dtype

    # Convert grid to homogeneous coordinates: (H, W, 3)
    grid_h = to_homogeneous(grid.to(device=device, dtype=dtype))

    # Reshape for batch matrix multiplication: (H*W, 3)
    grid_flat = grid_h.reshape(-1, 3)

    # Apply transformation: g @ grid_flat.T -> (B, 3, H*W)
    # Then transpose to (B, H*W, 3)
    transformed = torch.bmm(
        g,
        grid_flat.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)
    ).transpose(1, 2)

    # Convert back from homogeneous: (B, H*W, 2)
    transformed_2d = from_homogeneous(transformed)

    # Reshape to (B, H, W, 2)
    return transformed_2d.reshape(B, H, W, 2)


def warp_image(
    image: torch.Tensor,
    g: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True
) -> torch.Tensor:
    """
    Warp image by Sim(2) transformation using inverse warping.

    Implements: (g · I)(p) = I(g⁻¹ · p)

    For each destination pixel p, we compute g⁻¹ · p to find where
    to sample from the source image.

    Args:
        image: (B, C, H, W) source image tensor
        g: (B, 3, 3) Sim(2) transformation matrices
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        padding_mode: How to handle out-of-bounds samples
                     ('zeros', 'border', 'reflection')
        align_corners: If True, corner pixels are exactly at [-1, 1]

    Returns:
        (B, C, H, W) warped image tensor

    Note:
        The transformation g represents how to go from source to destination.
        We use g⁻¹ to find source coordinates for each destination pixel.
    """
    B, C, H, W = image.shape

    # Create normalized destination grid
    grid = create_normalized_grid(H, W, device=image.device, dtype=image.dtype)

    # Compute inverse transformation
    g_inv = Sim2.inverse(g)

    # For normalized coordinates, we need to adjust the transformation.
    # The Sim(2) matrix operates on some coordinate system.
    # We'll assume the transformation is defined in normalized coordinates.

    # Apply inverse transform to get source coordinates
    source_coords = apply_transform_to_grid(g_inv, grid)

    # Use grid_sample to warp
    # grid_sample expects (B, H, W, 2) with (x, y) in [-1, 1]
    warped = F.grid_sample(
        image,
        source_coords,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners
    )

    return warped


def warp_image_pixel_coords(
    image: torch.Tensor,
    g: torch.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros'
) -> torch.Tensor:
    """
    Warp image with transformation defined in pixel coordinates.

    This version assumes g operates on pixel coordinates:
        g · [x, y, 1]ᵀ maps pixel (x, y) to another pixel location

    Args:
        image: (B, C, H, W) source image tensor
        g: (B, 3, 3) Sim(2) matrices in pixel coordinate space
        mode: Interpolation mode
        padding_mode: Padding mode

    Returns:
        (B, C, H, W) warped image tensor
    """
    B, C, H, W = image.shape

    # Create pixel coordinate grid
    grid = create_pixel_grid(H, W, device=image.device, dtype=image.dtype)

    # Compute inverse transformation
    g_inv = Sim2.inverse(g)

    # Apply inverse transform in pixel coordinates
    source_coords_pixel = apply_transform_to_grid(g_inv, grid)

    # Convert to normalized coordinates for grid_sample
    source_coords_norm = pixel_to_normalized(source_coords_pixel, H, W)

    # Use grid_sample
    warped = F.grid_sample(
        image,
        source_coords_norm,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True
    )

    return warped


def transform_points(
    points: torch.Tensor,
    g: torch.Tensor
) -> torch.Tensor:
    """
    Transform 2D points by Sim(2) transformation.

    Args:
        points: (B, N, 2) or (N, 2) tensor of (x, y) coordinates
        g: (B, 3, 3) or (3, 3) transformation matrices

    Returns:
        Transformed points with same shape as input
    """
    # Handle unbatched inputs
    points_batched = points.unsqueeze(0) if points.dim() == 2 else points
    g_batched = g.unsqueeze(0) if g.dim() == 2 else g

    B, N, _ = points_batched.shape

    # Convert to homogeneous: (B, N, 3)
    points_h = to_homogeneous(points_batched)

    # Apply transformation: (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    transformed_h = torch.bmm(g_batched, points_h.transpose(1, 2)).transpose(1, 2)

    # Convert back: (B, N, 2)
    transformed = from_homogeneous(transformed_h)

    # Return with original batch dimension
    if points.dim() == 2:
        return transformed.squeeze(0)
    return transformed


def get_valid_mask(
    g: torch.Tensor,
    height: int,
    width: int,
    margin: float = 0.0
) -> torch.Tensor:
    """
    Compute mask of valid pixels after warping.

    A pixel is valid if its source location falls within the original image.

    Args:
        g: (B, 3, 3) Sim(2) transformation matrices
        height: Image height
        width: Image width
        margin: Extra margin to consider as invalid (in normalized coords)

    Returns:
        (B, H, W) boolean mask of valid pixels
    """
    B = g.shape[0]
    device = g.device
    dtype = g.dtype

    # Create normalized destination grid
    grid = create_normalized_grid(height, width, device=device, dtype=dtype)

    # Compute inverse transformation
    g_inv = Sim2.inverse(g)

    # Get source coordinates
    source_coords = apply_transform_to_grid(g_inv, grid)  # (B, H, W, 2)

    # Check if source coordinates are within [-1-margin, 1+margin]
    valid_x = (source_coords[..., 0] >= -1 - margin) & (source_coords[..., 0] <= 1 + margin)
    valid_y = (source_coords[..., 1] >= -1 - margin) & (source_coords[..., 1] <= 1 + margin)

    return valid_x & valid_y


def compute_overlap_ratio(
    g: torch.Tensor,
    height: int,
    width: int
) -> torch.Tensor:
    """
    Compute the overlap ratio after transformation.

    This measures what fraction of the destination image has valid source data.

    Args:
        g: (B, 3, 3) Sim(2) transformation matrices
        height: Image height
        width: Image width

    Returns:
        (B,) tensor of overlap ratios in [0, 1]
    """
    valid_mask = get_valid_mask(g, height, width)
    return valid_mask.float().mean(dim=(-2, -1))


def test_warping():
    """Test warping operations."""
    import torch

    print("Testing warping operations...")

    # Test 1: Identity transformation should not change image
    B, C, H, W = 2, 3, 64, 64
    image = torch.randn(B, C, H, W)
    g_id = Sim2.identity(B, device=image.device, dtype=image.dtype)

    warped = warp_image(image, g_id)
    assert torch.allclose(image, warped, atol=1e-4), "Identity warp should preserve image"
    print("✓ Identity warp preserves image")

    # Test 2: Inverse warp should (approximately) recover original
    xi = torch.tensor([[0.1, 0.05, 0.1, -0.1],
                       [-0.1, -0.05, -0.05, 0.1]])
    g = Sim2.exp(xi)
    g_inv = Sim2.inverse(g)

    warped = warp_image(image, g)
    recovered = warp_image(warped, g_inv)

    # Check center region (edges may have boundary effects)
    margin = 10
    center_orig = image[:, :, margin:-margin, margin:-margin]
    center_recov = recovered[:, :, margin:-margin, margin:-margin]
    assert torch.allclose(center_orig, center_recov, atol=0.1), "Double warp should recover center"
    print("✓ Inverse warp approximately recovers image")

    # Test 3: Point transformation
    points = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]]])  # (1, 3, 2)
    g = Sim2.identity(1)

    transformed = transform_points(points, g)
    assert torch.allclose(points, transformed, atol=1e-6), "Identity should preserve points"
    print("✓ Identity preserves points")

    # Test 4: Point transformation with rotation
    theta = torch.tensor([torch.pi / 2])  # 90 degrees
    scale = torch.tensor([1.0])
    trans = torch.tensor([[0.0, 0.0]])
    g_rot = Sim2.from_components(theta, scale, trans)

    point = torch.tensor([[[1.0, 0.0]]])  # Point on x-axis
    rotated = transform_points(point, g_rot)
    expected = torch.tensor([[[0.0, 1.0]]])  # Should be on y-axis
    assert torch.allclose(rotated, expected, atol=1e-5), "90° rotation should map (1,0) to (0,1)"
    print("✓ Rotation transforms points correctly")

    # Test 5: Gradient flow through warping
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    xi = torch.tensor([[0.1, 0.05, 0.1, -0.1]], requires_grad=True)
    g = Sim2.exp(xi)

    warped = warp_image(image, g)
    loss = warped.sum()
    loss.backward()

    assert image.grad is not None, "Gradients should flow to image"
    assert xi.grad is not None, "Gradients should flow to transformation"
    print("✓ Gradients flow through warping")

    # Test 6: Valid mask
    g = Sim2.identity(1)
    mask = get_valid_mask(g, 32, 32)
    assert mask.all(), "Identity should have all valid pixels"
    print("✓ Valid mask correct for identity")

    # Test 7: Overlap ratio
    g = Sim2.identity(1)
    overlap = compute_overlap_ratio(g, 32, 32)
    assert torch.allclose(overlap, torch.tensor([1.0])), "Identity should have 100% overlap"
    print("✓ Overlap ratio correct for identity")

    print("\nAll warping tests passed!")


if __name__ == "__main__":
    test_warping()
