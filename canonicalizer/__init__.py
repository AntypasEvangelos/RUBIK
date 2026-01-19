"""
Multiview Canonicalizer for Dense Image Matching

This module implements a conditional multiview canonicalizer that preprocesses
image pairs before feeding them to any dense image matcher. The goal is to
improve matching performance on geometrically challenging scenarios by factoring
out viewpoint, scale, and translation differences via a learned Sim(2) transformation.
"""

from .core.lie_group import Sim2
from .core.warping import warp_image, create_normalized_grid
from .core.energy import cosine_distance, l2_distance

__version__ = "0.1.0"

__all__ = [
    "Sim2",
    "warp_image",
    "create_normalized_grid",
    "cosine_distance",
    "l2_distance",
]
