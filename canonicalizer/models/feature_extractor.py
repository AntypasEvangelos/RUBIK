
import torch
import torch.nn as nn
from typing import Dict, Any

class DINOv2FeatureExtractor(nn.Module):
    """
    Wrapper for DINOv2 feature extractor.
    
    Loads DINOv2 from torch.hub and provides methods to extract features.
    """
    def __init__(self, model_name: str = 'dinov2_vitb14', freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input images.

        Args:
            x: (B, C, H, W) input images
               - If already normalized with ImageNet mean/std, pass directly
               - If in [0, 1] range, will be normalized automatically
               - If in [0, 255] range, will be converted and normalized automatically

        Returns:
            Dictionary containing 'x_norm_patchtokens' and other features
        """
        # Normalize input if not already normalized
        # Check if input is in [0, 255] range
        if x.max() > 2.0:  # Assume [0, 255] range
            x = x / 255.0

        # Check if input is in [0, 1] range (not yet normalized with mean/std)
        # If already normalized, mean should be close to 0, std close to 1
        if x.min() >= 0.0 and x.max() <= 1.0:
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std

        # DINOv2 forward_features returns a dict
        return self.model.forward_features(x)
