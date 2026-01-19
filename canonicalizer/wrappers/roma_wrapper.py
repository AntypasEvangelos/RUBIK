
import torch
import torch.nn as nn
from typing import Dict, Any

from .base_wrapper import CanonicalizerWrapper

class RoMaWrapper(CanonicalizerWrapper):
    """
    Wrapper specifically for the RoMa matcher.
    """
    def __init__(self, canonicalizer, roma_model):
        super().__init__(canonicalizer, roma_model)
        
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        RoMa specific forward pass adaptation.
        """
        # RoMa typically expects a forward method taking dict or explicit batch
        # We reuse the base wrapper logic but ensure RoMa's output format is handled
        return super().forward(data)

    # If RoMa has specific matching API (e.g. match() instead of forward())
    # we can expose it here.
    def match(self, image0, image1, **kwargs):
        data = {'image0': image0, 'image1': image1, **kwargs}
        return self.forward(data)
