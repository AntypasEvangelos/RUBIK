
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..core.lie_group import Sim2

class CanonicalizerWrapper(nn.Module):
    """
    Wraps any image matcher with canonicalization preprocessing.
    
    Pipeline:
    1. Canonicalize source image (align to target).
    2. Match (Warped Source, Target).
    3. Transform keypoints back to original Source frame.
    """
    def __init__(
        self,
        canonicalizer: nn.Module,
        matcher: Any,
        transform_matches_back: bool = True
    ):
        super().__init__()
        self.canonicalizer = canonicalizer
        self.matcher = matcher
        self.transform_back = transform_matches_back
        
    def forward(
        self, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward pass handling dictionary-based inputs/outputs common in matching.
        
        Args:
            data: Dictionary containing:
                - 'image0': Source image (B, C, H, W)
                - 'image1': Target image (B, C, H, W)
                - ... other matcher args
        Returns:
            Dictionary with matches and metadata
        """
        source = data['image0']
        target = data['image1']
        
        # 1. Canonicalize
        warped_source, g_star = self.canonicalizer(source, target)
        
        # Update data for matcher
        # We probably shouldn't modify the original dict in place if we can avoid it,
        # but many matchers expect specific keys.
        matcher_input = data.copy()
        matcher_input['image0'] = warped_source
        
        # 2. Match
        # Matcher is expected to return a dict with 'keypoints0', 'keypoints1', etc.
        # or we might need specific adapter logic.
        matches = self.matcher(matcher_input)
        
        # 3. Transform back
        if self.transform_back and 'keypoints0' in matches:
            kpts0 = matches['keypoints0'] # (B, N, 2)
            
            # g_star maps Source -> Target (Canonical)
            # So Warped = g_star · Source
            # kpts0 are in Warped frame
            # We need kpts0_orig in Source frame
            # Warped_pts = g_star · Source_pts
            # Source_pts = g_star⁻¹ · Warped_pts
            
            g_inv = Sim2.inverse(g_star)
            
            # Use Sim2 transformation utility
            from ..core.warping import transform_points
            kpts0_orig = transform_points(kpts0, g_inv)
            
            matches['keypoints0'] = kpts0_orig
            
        # Store canonicalization info
        matches['canonical_transform'] = g_star
        matches['warped_source'] = warped_source
        
        return matches
