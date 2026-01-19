"""
Core mathematical components for the canonicalizer.

- lie_group: Sim(2) Lie group operations (exp, log, compose, inverse)
- warping: Differentiable image warping via grid_sample
- energy: Energy/distance functions for feature comparison
"""

from .lie_group import Sim2
from .warping import warp_image, create_normalized_grid, transform_points
from .energy import cosine_distance, l2_distance, EnergyFunction

__all__ = [
    "Sim2",
    "warp_image",
    "create_normalized_grid",
    "transform_points",
    "cosine_distance",
    "l2_distance",
    "EnergyFunction",
]
