
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Optional

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
    return (img * 255).astype(np.uint8)

def visualize_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    warped: torch.Tensor,
    kpts0: torch.Tensor = None,
    kpts1: torch.Tensor = None,
    max_items: int = 4
) -> Dict[str, np.ndarray]:
    """
    Create visualizations for a batch of data.
    
    Args:
        source: (B, C, H, W) source images
        target: (B, C, H, W) target images
        warped: (B, C, H, W) warped source images
        kpts0: (B, N, 2) keypoints in source/warped frame
        kpts1: (B, N, 2) keypoints in target frame
        
    Returns:
        Dict mapping names to numpy images/plots
    """
    B = source.shape[0]
    n = min(B, max_items)
    
    vis_images = []
    
    for i in range(n):
        img_s = tensor_to_image(source[i])
        img_t = tensor_to_image(target[i])
        img_w = tensor_to_image(warped[i])
        
        # Concatenate horizontally: Source | Warped | Target
        H, W, C = img_s.shape
        combined = np.hstack([img_s, img_w, img_t])
        
        # Draw keypoints if available
        if kpts0 is not None and kpts1 is not None and kpts0.shape[1] > 0:
            kp0 = kpts0[i].detach().cpu().numpy()
            kp1 = kpts1[i].detach().cpu().numpy()
            
            # Simple drawing
            combined = combined.copy()
            # Draw on Warped (middle) and Target (right)
            offset_w = W
            offset_t = 2*W
            
            # Draw lines? Too messy for dense points. Draw small circles.
            for k in range(min(50, len(kp0))): # Limit to 50 points
                x0, y0 = int(kp0[k, 0]) + offset_w, int(kp0[k, 1])
                x1, y1 = int(kp1[k, 0]) + offset_t, int(kp1[k, 1])
                
                # Check bounds
                if 0 <= y0 < H and 0 <= x0 < 3*W and 0 <= y1 < H and 0 <= x1 < 3*W:
                    color = (0, 255, 0)
                    cv2.circle(combined, (x0, y0), 3, color, -1)
                    cv2.circle(combined, (x1, y1), 3, color, -1)
                    cv2.line(combined, (x0, y0), (x1, y1), (0, 0, 255), 1)

        vis_images.append(combined)
        
    # Stack vertically
    final_vis = np.vstack(vis_images)
    
    return {"warps": final_vis}
