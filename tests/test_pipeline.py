
import torch
import sys
import os

# Add parent directory to path so we can import canonicalizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer
from canonicalizer.core.lie_group import Sim2

def test_frozen_energy_pipeline():
    print("Testing FrozenEnergyCanonicalizer pipeline...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Instantiate model
    # Use 'dinov2_vits14' (small) for faster test or 'dinov2_vitb14' (base)
    try:
        model = FrozenEnergyCanonicalizer(
            dino_model='dinov2_vits14', 
            num_iterations=3,
            step_size=0.1
        ).to(device)
        print("✓ Model instantiated")
    except Exception as e:
        print(f"✕ Model instantiation failed: {e}")
        # Assuming internet might be an issue for torch.hub, but let's see.
        return

    # Create dummy data
    B, C, H, W = 2, 3, 224, 224 # DINO likes multiples of 14
    source = torch.randn(B, C, H, W).to(device)
    target = torch.randn(B, C, H, W).to(device)
    
    # Run forward pass
    try:
        warped, g_star = model(source, target)
        print("✓ Forward pass completed")
    except Exception as e:
        print(f"✕ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Check outputs
    if warped.shape != source.shape:
        print(f"✕ Output shape mismatch: expected {source.shape}, got {warped.shape}")
    else:
        print("✓ Output shape correct")
        
    if g_star.shape != (B, 3, 3):
        print(f"✕ Transform shape mismatch: expected {(B, 3, 3)}, got {g_star.shape}")
    else:
        print("✓ Transform shape correct")
        
    # Check if backprop works (training mode)
    model.train()
    loss = warped.mean() + g_star.mean()
    try:
        loss.backward()
        print("✓ Gradients flow check passed")
    except Exception as e:
        print(f"✕ Backprop failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frozen_energy_pipeline()
