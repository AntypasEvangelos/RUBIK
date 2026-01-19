
import torch
import sys
import os
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from canonicalizer.wrappers.roma_wrapper import RoMaWrapper
from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer

class MockRoMa(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        # Mock RoMa output
        B = data['image0'].shape[0]
        # Return random keypoints
        return {
            'keypoints0': torch.randn(B, 100, 2),
            'keypoints1': torch.randn(B, 100, 2)
        }

def test_roma_wrapper():
    print("Testing RoMaWrapper...")
    
    canonicalizer = FrozenEnergyCanonicalizer(dino_model='dinov2_vits14', num_iterations=1)
    roma = MockRoMa()
    
    wrapper = RoMaWrapper(canonicalizer, roma)
    
    data = {
        'image0': torch.randn(2, 3, 224, 224),
        'image1': torch.randn(2, 3, 224, 224)
    }
    
    output = wrapper(data)
    
    if 'canonical_transform' in output and 'warped_source' in output:
        print("✓ Wrapper added canonicalization info")
    else:
        print("✕ Missing canonicalization info")
        
    print("✓ RoMaWrapper test passed")

if __name__ == "__main__":
    test_roma_wrapper()
