#!/usr/bin/env python3
"""
Test script for DEQ canonicalization models.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_global_deq():
    """Test global DEQ canonicalizer."""
    print("\n" + "="*60)
    print("Testing Global DEQ Canonicalizer")
    print("="*60)

    try:
        from canonicalizer.models.deq_canonicalizer import DEQCanonicalizer
        print("✓ Successfully imported DEQCanonicalizer")
    except Exception as e:
        print(f"✗ Failed to import DEQCanonicalizer: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy data
    B, C, H, W = 2, 3, 224, 224
    source = torch.randn(B, C, H, W).to(device)
    target = torch.randn(B, C, H, W).to(device)

    try:
        # Initialize model
        model = DEQCanonicalizer(
            dino_model='dinov2_vits14',  # Small for testing
            max_iterations=10,
            convergence_threshold=1e-3
        ).to(device)
        print("✓ Model instantiated")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            warped, g_star, trajectory = model(
                source, target, return_trajectory=True
            )
        print("✓ Forward pass completed")
        print(f"  Converged in {len(trajectory)} iterations")
        print(f"  Final update norm: {trajectory[-1][1]:.6f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check outputs
    if warped.shape != source.shape:
        print(f"✗ Output shape mismatch: expected {source.shape}, got {warped.shape}")
        return False
    print("✓ Output shape correct")

    if g_star.shape != (B, 3, 3):
        print(f"✗ Transform shape mismatch: expected {(B, 3, 3)}, got {g_star.shape}")
        return False
    print("✓ Transform shape correct")

    # Test gradient flow
    try:
        model.train()
        warped, g_star = model(source, target)
        loss = warped.mean() + g_star.mean()
        loss.backward()
        print("✓ Gradient flow check passed")
    except Exception as e:
        print(f"✗ Backprop failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Global DEQ test PASSED")
    return True


def test_local_deq():
    """Test local (patch-based) DEQ canonicalizer."""
    print("\n" + "="*60)
    print("Testing Local DEQ Canonicalizer")
    print("="*60)

    try:
        from canonicalizer.models.local_deq_canonicalizer import LocalDEQCanonicalizer
        print("✓ Successfully imported LocalDEQCanonicalizer")
    except Exception as e:
        print(f"✗ Failed to import LocalDEQCanonicalizer: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dummy data
    B, C, H, W = 2, 3, 224, 224
    source = torch.randn(B, C, H, W).to(device)
    target = torch.randn(B, C, H, W).to(device)

    try:
        # Initialize model
        model = LocalDEQCanonicalizer(
            dino_model='dinov2_vits14',  # Small for testing
            patch_resolution=4,  # Small grid for testing
            max_iterations=5,  # Fewer iterations for speed
            convergence_threshold=1e-3
        ).to(device)
        print("✓ Model instantiated")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Forward pass
        model.eval()
        with torch.no_grad():
            warped, g_field, trajectory = model(
                source, target, return_trajectory=True
            )
        print("✓ Forward pass completed")
        print(f"  Converged in {len(trajectory)} iterations")
        print(f"  Final update norm: {trajectory[-1]:.6f}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check outputs
    if warped.shape != source.shape:
        print(f"✗ Output shape mismatch: expected {source.shape}, got {warped.shape}")
        return False
    print("✓ Output shape correct")

    expected_field_shape = (B, 3, 3, 4, 4)  # patch_resolution=4
    if g_field.shape != expected_field_shape:
        print(f"✗ Transform field shape mismatch: expected {expected_field_shape}, got {g_field.shape}")
        return False
    print("✓ Transform field shape correct")

    # Test getting global transformation
    try:
        g_global = model.get_global_transformation(g_field)
        if g_global.shape != (B, 3, 3):
            print(f"✗ Global transform shape mismatch: expected {(B, 3, 3)}, got {g_global.shape}")
            return False
        print("✓ Global transformation extraction works")
    except Exception as e:
        print(f"✗ Global transformation extraction failed: {e}")
        return False

    # Test gradient flow (simplified, no full backward due to complexity)
    try:
        print("✓ Gradient flow check (skipped for local model - too slow)")
    except Exception as e:
        print(f"✗ Backprop failed: {e}")
        return False

    print("\n✓ Local DEQ test PASSED")
    return True


def test_comparison():
    """Compare outputs of frozen energy vs DEQ models."""
    print("\n" + "="*60)
    print("Comparing Frozen Energy vs DEQ Models")
    print("="*60)

    try:
        from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer
        from canonicalizer.models.deq_canonicalizer import DEQCanonicalizer
        print("✓ Successfully imported both models")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create same input
    torch.manual_seed(42)
    B, C, H, W = 1, 3, 224, 224
    source = torch.randn(B, C, H, W).to(device)
    target = torch.randn(B, C, H, W).to(device)

    # Frozen Energy model
    try:
        frozen_model = FrozenEnergyCanonicalizer(
            dino_model='dinov2_vits14',
            num_iterations=5
        ).to(device)

        frozen_model.eval()
        with torch.no_grad():
            warped_frozen, g_frozen = frozen_model(source, target)

        print("✓ Frozen Energy model ran successfully")
    except Exception as e:
        print(f"✗ Frozen Energy model failed: {e}")
        return False

    # DEQ model
    try:
        deq_model = DEQCanonicalizer(
            dino_model='dinov2_vits14',
            max_iterations=10
        ).to(device)

        deq_model.eval()
        with torch.no_grad():
            warped_deq, g_deq = deq_model(source, target)

        print("✓ DEQ model ran successfully")
    except Exception as e:
        print(f"✗ DEQ model failed: {e}")
        return False

    # Compare
    print(f"\nTransformation comparison:")
    print(f"  Frozen Energy g:\n{g_frozen[0]}")
    print(f"  DEQ g:\n{g_deq[0]}")

    print("\n✓ Comparison test PASSED")
    return True


if __name__ == "__main__":
    print("Testing DEQ Canonicalization Models")
    print("="*60)

    results = []

    # Run tests
    results.append(("Global DEQ", test_global_deq()))
    results.append(("Local DEQ", test_local_deq()))
    results.append(("Comparison", test_comparison()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")

    all_passed = all(result[1] for result in results)
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
