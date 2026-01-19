
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
            x: (B, C, H, W) input images, normalized to [0, 1] then DINO mean/std
            
        Returns:
            Dictionary containing 'x_norm_patchtokens' and other features
        """
        # DINOv2 forward_features returns a dict
        return self.model.forward_features(x)
