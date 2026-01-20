
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, Optional
import torch.nn.functional as F

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert (C, H, W) tensor to (H, W, C) numpy image.
    Un-normalizes ImageNet mean/std.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def flow_to_color(flow: np.ndarray, max_flow: float = None) -> np.ndarray:
    """
    Convert optical flow to color visualization (HSV color wheel).

    Args:
        flow: (H, W, 2) flow field (dx, dy)
        max_flow: Maximum flow magnitude for normalization

    Returns:
        (H, W, 3) RGB image
    """
    H, W = flow.shape[:2]

    # Compute flow magnitude and angle
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)

    # Normalize magnitude
    if max_flow is None:
        max_flow = mag.max()

    mag = np.clip(mag / (max_flow + 1e-8), 0, 1)

    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi) * 179  # Hue: angle
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = (mag * 255).astype(np.uint8)  # Value: magnitude

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return rgb

def warp_to_flow(warp: torch.Tensor) -> np.ndarray:
    """
    Convert RoMa-style warp to dense flow field.

    Args:
        warp: (B, H, W, 4) warp from RoMa (x, y, certainty, ?)
              or (B, H, W, 2) dense correspondence map

    Returns:
        (H, W, 2) flow field (dx, dy)
    """
    if warp.shape[-1] == 4:
        warp = warp[..., :2]  # Extract (x, y) only

    # Convert first item in batch
    warp = warp[0].detach().cpu().numpy()  # (H, W, 2)

    # Create identity grid
    H, W = warp.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    identity = np.stack([xx, yy], axis=-1).astype(np.float32)

    # Flow = warp - identity
    flow = warp - identity

    return flow

def visualize_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    warped: torch.Tensor,
    kpts0: torch.Tensor = None,
    kpts1: torch.Tensor = None,
    matcher_warp: torch.Tensor = None,
    max_items: int = 4,
    dpi: int = 100
) -> Dict[str, np.ndarray]:
    """
    Create comprehensive visualizations for a batch of data.

    Args:
        source: (B, C, H, W) source images
        target: (B, C, H, W) target images
        warped: (B, C, H, W) warped source images (after canonicalization)
        kpts0: (B, N, 2) keypoints in source/warped frame (optional)
        kpts1: (B, N, 2) keypoints in target frame (optional)
        matcher_warp: (B, H, W, 2 or 4) dense warp from matcher (optional)
        max_items: Maximum number of batch items to visualize
        dpi: DPI for matplotlib figures

    Returns:
        Dict mapping names to numpy images/plots
    """
    B = source.shape[0]
    n = min(B, max_items)

    results = {}

    # ===== 1. Basic warp visualization with titles =====
    for i in range(n):
        img_s = tensor_to_image(source[i])
        img_t = tensor_to_image(target[i])
        img_w = tensor_to_image(warped[i])

        H, W, C = img_s.shape

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

        axes[0].imshow(img_s)
        axes[0].set_title('Source Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img_w)
        axes[1].set_title('Canonicalized Source\n(After Canonicalizer)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(img_t)
        axes[2].set_title('Target Image', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # Draw keypoints if available
        if kpts0 is not None and kpts1 is not None and kpts0.shape[1] > 0:
            kp0 = kpts0[i].detach().cpu().numpy()
            kp1 = kpts1[i].detach().cpu().numpy()

            # Draw on warped (middle) and target (right)
            num_pts = min(50, len(kp0))  # Limit to 50 points
            for k in range(num_pts):
                x0, y0 = kp0[k, 0], kp0[k, 1]
                x1, y1 = kp1[k, 0], kp1[k, 1]

                # Check bounds
                if 0 <= y0 < H and 0 <= x0 < W and 0 <= y1 < H and 0 <= x1 < W:
                    axes[1].plot(x0, y0, 'go', markersize=4)
                    axes[2].plot(x1, y1, 'go', markersize=4)

        plt.tight_layout()

        # Convert figure to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        results[f'warp_{i}'] = img_array
        plt.close(fig)

    # Stack all warp visualizations vertically
    if n > 0:
        warp_images = [results[f'warp_{i}'] for i in range(n)]
        results['warps'] = np.vstack(warp_images)

    # ===== 2. Matcher warp visualization (if provided) =====
    if matcher_warp is not None:
        for i in range(n):
            # Convert matcher warp to flow
            if len(matcher_warp.shape) == 4:  # (B, H, W, 2 or 4)
                flow = warp_to_flow(matcher_warp[i:i+1])
            else:
                continue

            # Visualize flow as color
            flow_color = flow_to_color(flow)

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

            img_w = tensor_to_image(warped[i])
            img_t = tensor_to_image(target[i])

            axes[0].imshow(img_w)
            axes[0].set_title('Canonicalized Source', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(flow_color)
            axes[1].set_title('Matcher Dense Flow\n(Canonicalized → Target)', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(img_t)
            axes[2].set_title('Target Image', fontsize=14, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()

            # Convert to numpy
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            results[f'matcher_flow_{i}'] = img_array
            plt.close(fig)

        # Stack all matcher flow visualizations
        if n > 0:
            flow_images = [results[f'matcher_flow_{i}'] for i in range(n)]
            results['matcher_flows'] = np.vstack(flow_images)

    # ===== 3. Side-by-side comparison =====
    if n > 0:
        fig, axes = plt.subplots(n, 3, figsize=(15, 5*n), dpi=dpi, squeeze=False)

        for i in range(n):
            img_s = tensor_to_image(source[i])
            img_w = tensor_to_image(warped[i])
            img_t = tensor_to_image(target[i])

            axes[i, 0].imshow(img_s)
            axes[i, 0].set_title('Source' if i == 0 else '', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(img_w)
            axes[i, 1].set_title('Canonicalized' if i == 0 else '', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(img_t)
            axes[i, 2].set_title('Target' if i == 0 else '', fontsize=12, fontweight='bold')
            axes[i, 2].axis('off')

        plt.tight_layout()

        # Convert to numpy
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        results['comparison'] = img_array
        plt.close(fig)

    return results


def visualize_matcher_output(
    source: torch.Tensor,
    target: torch.Tensor,
    warped: torch.Tensor,
    matcher_output: Dict,
    max_items: int = 4,
    dpi: int = 100
) -> Dict[str, np.ndarray]:
    """
    Visualize matcher output including dense warp and matches.

    Args:
        source: (B, C, H, W) source images
        target: (B, C, H, W) target images
        warped: (B, C, H, W) canonicalized source
        matcher_output: Dictionary from matcher containing 'warp', 'certainty', etc.
        max_items: Maximum batch items to visualize
        dpi: DPI for figures

    Returns:
        Dict of visualizations
    """
    # Extract matcher warp if available
    matcher_warp = matcher_output.get('warp', None)

    # Call main visualization function
    return visualize_batch(
        source=source,
        target=target,
        warped=warped,
        matcher_warp=matcher_warp,
        max_items=max_items,
        dpi=dpi
    )
