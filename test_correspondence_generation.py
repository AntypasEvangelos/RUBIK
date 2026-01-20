#!/usr/bin/env python3
"""
Test script to verify correspondence generation from depth maps.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_correspondence_generation():
    """Test that we can generate correspondences from a sample in the dataset."""

    print("Testing correspondence generation...")

    try:
        from canonicalizer.training.datasets import RubikDataset
        print("✓ Successfully imported RubikDataset")
    except Exception as e:
        print(f"✗ Failed to import RubikDataset: {e}")
        return False

    # Paths (update these if needed)
    data_path = "rubik.json"
    nuscenes_path = "/vast/projects/kostas/geometric-learning/nuscenes/nuscenes-download"
    unidepths_path = "/vast/projects/kostas/geometric-learning/nuscenes/unidepths"

    # Check if paths exist
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("  (This is expected if running outside RUBIK root)")
        return False

    if not os.path.exists(nuscenes_path):
        print(f"✗ NuScenes path not found: {nuscenes_path}")
        print("  (This test requires access to the full dataset)")
        return False

    if not os.path.exists(unidepths_path):
        print(f"✗ Unidepths path not found: {unidepths_path}")
        print("  (This test requires depth maps)")
        return False

    try:
        # Initialize dataset
        print("\nInitializing dataset...")
        dataset = RubikDataset(
            data_path=data_path,
            nuscenes_path=nuscenes_path,
            unidepths_path=unidepths_path,
            mode='train',
            img_size=224,  # Smaller for faster test
            num_correspondences=100
        )
        print(f"✓ Dataset initialized with {len(dataset)} samples")

    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        # Load first sample
        print("\nLoading first sample...")
        sample = dataset[0]
        print("✓ Sample loaded successfully")

        # Check outputs
        print("\nChecking sample contents:")
        print(f"  image0 shape: {sample['image0'].shape}")
        print(f"  image1 shape: {sample['image1'].shape}")
        print(f"  keypoints0 shape: {sample['keypoints0'].shape}")
        print(f"  keypoints1 shape: {sample['keypoints1'].shape}")
        print(f"  pose shape: {sample['pose'].shape}")
        print(f"  K0 shape: {sample['K0'].shape}")
        print(f"  K1 shape: {sample['K1'].shape}")

        # Verify shapes
        assert sample['image0'].shape[0] == 3, "Image should have 3 channels"
        assert sample['keypoints0'].shape[1] == 2, "Keypoints should be 2D"
        assert sample['keypoints1'].shape[1] == 2, "Keypoints should be 2D"
        assert sample['keypoints0'].shape[0] > 0, "Should have generated correspondences"
        assert sample['keypoints0'].shape[0] == sample['keypoints1'].shape[0], "Should have same number of keypoints"

        print(f"\n✓ Generated {sample['keypoints0'].shape[0]} correspondences")

        # Check that keypoints are in normalized coordinates [-1, 1]
        kpts0 = sample['keypoints0']
        kpts1 = sample['keypoints1']

        print(f"\nKeypoint coordinate ranges:")
        print(f"  keypoints0: x=[{kpts0[:, 0].min():.3f}, {kpts0[:, 0].max():.3f}], y=[{kpts0[:, 1].min():.3f}, {kpts0[:, 1].max():.3f}]")
        print(f"  keypoints1: x=[{kpts1[:, 0].min():.3f}, {kpts1[:, 0].max():.3f}], y=[{kpts1[:, 1].min():.3f}, {kpts1[:, 1].max():.3f}]")

        # Should be roughly in [-1, 1] range
        assert kpts0.min() >= -1.5 and kpts0.max() <= 1.5, "Keypoints should be in normalized range"
        assert kpts1.min() >= -1.5 and kpts1.max() <= 1.5, "Keypoints should be in normalized range"

        print("✓ Keypoints are in normalized coordinate range")

    except Exception as e:
        print(f"✗ Failed to load/verify sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_correspondence_generation()
    sys.exit(0 if success else 1)
