
import torch
from torch.utils.data import Dataset
import json
import os
import os.path as osp
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
import random

class RubikDataset(Dataset):
    """
    Dataset loader for RUBIK/nuScenes training data.

    Generates GT correspondences from depth maps and GT pose for supervision.
    This aligns with RUBIK's evaluation: pose estimation from matches.
    """
    def __init__(
        self,
        data_path: str,
        nuscenes_path: str,
        unidepths_path: str,
        mode: str = 'train',
        transform = None,
        img_size: int = 518,  # DINOv2 friendly size
        num_correspondences: int = 500  # Number of GT correspondences to generate
    ):
        self.data = json.load(open(data_path))
        self.nuscenes_path = nuscenes_path
        self.unidepths_path = unidepths_path
        self.mode = mode
        self.num_correspondences = num_correspondences
        
        # Flatten the data structure: box -> scene -> pairs
        self.samples = []
        for box in self.data:
            for scene in self.data[box]:
                pairs = self.data[box][scene]
                for pair_key in pairs:
                    # pair_key is string representation of tuple: "('cam_front...jpg', 'cam_back...jpg')"
                    # We need to parse it back
                    pair_tuple = eval(pair_key)
                    
                    info = pairs[pair_key]
                    sample = {
                        'path0': pair_tuple[0],
                        'path1': pair_tuple[1],
                        'box': box,
                        'scene': scene,
                        'K1': np.array(info['K1']),
                        'K2': np.array(info['K2']),
                        'rel_pose': np.array(info['rel_pose'])
                    }
                    self.samples.append(sample)
                    
        # Simple split logic (e.g. 80/20 based on scenes or index)
        # For a benchmark dataset, usually test set is fixed. 
        # Assuming rubik.json is the full benchmark, we might just train on a subset or 
        # need a separate train split. For now, we take 90% as train if mode=train
        
        # Consistent shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split = int(0.9 * len(self.samples))
        
        if mode == 'train':
            self.indices = indices[:split]
        else:
            self.indices = indices[split:]
            
        self.img_size = img_size
        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.indices)

    def generate_correspondences(
        self,
        depth0: np.ndarray,
        depth1: np.ndarray,
        K0: np.ndarray,
        K1: np.ndarray,
        pose: np.ndarray,
        H0: int,
        W0: int,
        H1: int,
        W1: int,
        num_points: int = 500
    ):
        """
        Generate GT correspondences from depth maps and GT pose.

        This follows RUBIK's evaluation pipeline:
        1. Backproject source pixels to 3D using depth
        2. Transform to target frame using GT pose
        3. Project to target image

        Args:
            depth0: (H0, W0) depth map for source image
            depth1: (H1, W1) depth map for target image
            K0: (3, 3) intrinsics for source
            K1: (3, 3) intrinsics for target
            pose: (4, 4) relative pose [R|t; 0 0 0 1] from source to target
            H0, W0: Source image dimensions
            H1, W1: Target image dimensions
            num_points: Number of correspondences to generate

        Returns:
            pts0: (N, 2) keypoints in source image (pixel coords)
            pts1: (N, 2) keypoints in target image (pixel coords)
        """
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Sample random pixels from source image
        # Avoid borders and areas with invalid depth
        valid_mask = (depth0 > 0.1) & (depth0 < 100.0)  # Valid depth range
        valid_y, valid_x = np.where(valid_mask)

        if len(valid_y) < num_points:
            # Not enough valid points, return what we have
            num_points = len(valid_y)

        # Randomly sample from valid pixels
        indices = np.random.choice(len(valid_y), size=num_points, replace=False)
        sample_x = valid_x[indices]
        sample_y = valid_y[indices]

        # Get depths at sampled locations
        sample_depths = depth0[sample_y, sample_x]

        # Backproject to 3D (source camera frame)
        # [X, Y, Z]^T = depth * K^-1 @ [x, y, 1]^T
        K0_inv = np.linalg.inv(K0)

        pts_2d_0 = np.stack([sample_x, sample_y, np.ones_like(sample_x)], axis=1)  # (N, 3)
        pts_3d_0 = (K0_inv @ pts_2d_0.T).T  # (N, 3)
        pts_3d_0 = pts_3d_0 * sample_depths[:, np.newaxis]  # Scale by depth

        # Transform to target camera frame
        pts_3d_1 = (R @ pts_3d_0.T).T + t  # (N, 3)

        # Project to target image
        pts_2d_1_h = (K1 @ pts_3d_1.T).T  # (N, 3) homogeneous
        pts_2d_1 = pts_2d_1_h[:, :2] / pts_2d_1_h[:, 2:3]  # (N, 2) normalized

        # Filter points that project outside target image or behind camera
        valid = (
            (pts_2d_1[:, 0] >= 0) & (pts_2d_1[:, 0] < W1) &
            (pts_2d_1[:, 1] >= 0) & (pts_2d_1[:, 1] < H1) &
            (pts_3d_1[:, 2] > 0.1)  # In front of camera
        )

        pts0 = np.stack([sample_x, sample_y], axis=1).astype(np.float32)  # (N, 2)
        pts1 = pts_2d_1.astype(np.float32)  # (N, 2)

        # Apply validity mask
        pts0 = pts0[valid]
        pts1 = pts1[valid]

        return pts0, pts1

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.samples[real_idx]

        # Construct full paths
        # RUBIK/eval.py logic:
        # osp.join(nuscenes_path, "sweeps", el[0].split("__")[1].split("__")[0], el[0])
        def get_full_path(rel_path):
            sensor = rel_path.split("__")[1].split("__")[0]
            return osp.join(self.nuscenes_path, "sweeps", sensor, rel_path)

        path0 = get_full_path(sample['path0'])
        path1 = get_full_path(sample['path1'])

        # Load images
        # DINO requires RGB
        img0 = Image.open(path0).convert('RGB')
        img1 = Image.open(path1).convert('RGB')

        W0, H0 = img0.size
        W1, H1 = img1.size

        # Load depth maps
        # Depth maps are stored with same basename but .npy extension
        depth_path0 = osp.join(self.unidepths_path, osp.basename(path0).replace('.jpg', '.npy'))
        depth_path1 = osp.join(self.unidepths_path, osp.basename(path1).replace('.jpg', '.npy'))

        depth0 = np.load(depth_path0)
        depth1 = np.load(depth_path1)

        # Generate GT correspondences from depth + pose
        # This is the RUBIK-aligned supervision:
        # We generate point correspondences that the canonicalizer should align
        pts0_pixel, pts1_pixel = self.generate_correspondences(
            depth0=depth0,
            depth1=depth1,
            K0=sample['K1'],  # Note: K1 corresponds to image0
            K1=sample['K2'],  # Note: K2 corresponds to image1
            pose=sample['rel_pose'],
            H0=H0, W0=W0,
            H1=H1, W1=W1,
            num_points=self.num_correspondences
        )

        # Transform images (this will resize to img_size x img_size)
        if self.transform:
            img0_tensor = self.transform(img0)
            img1_tensor = self.transform(img1)

        # Scale keypoints to match resized images
        # Original size: (W0, H0) -> Resized: (img_size, img_size)
        scale_x0 = self.img_size / W0
        scale_y0 = self.img_size / H0
        scale_x1 = self.img_size / W1
        scale_y1 = self.img_size / H1

        pts0_scaled = pts0_pixel.copy()
        pts0_scaled[:, 0] *= scale_x0
        pts0_scaled[:, 1] *= scale_y0

        pts1_scaled = pts1_pixel.copy()
        pts1_scaled[:, 0] *= scale_x1
        pts1_scaled[:, 1] *= scale_y1

        # Convert to normalized coordinates [-1, 1] (g_star operates in normalized space)
        # Normalized: x_norm = 2 * (x / W) - 1
        def pixel_to_normalized(pts, H, W):
            pts_norm = pts.copy()
            pts_norm[:, 0] = 2.0 * (pts[:, 0] / W) - 1.0
            pts_norm[:, 1] = 2.0 * (pts[:, 1] / H) - 1.0
            return pts_norm

        pts0_norm = pixel_to_normalized(pts0_scaled, self.img_size, self.img_size)
        pts1_norm = pixel_to_normalized(pts1_scaled, self.img_size, self.img_size)

        return {
            'image0': img0_tensor,
            'image1': img1_tensor,
            'keypoints0': torch.from_numpy(pts0_norm).float(),
            'keypoints1': torch.from_numpy(pts1_norm).float(),
            'pose': torch.tensor(sample['rel_pose'], dtype=torch.float32),
            'K0': torch.tensor(sample['K1'], dtype=torch.float32),
            'K1': torch.tensor(sample['K2'], dtype=torch.float32),
            'original_size0': torch.tensor([H0, W0], dtype=torch.float32),
            'original_size1': torch.tensor([H1, W1], dtype=torch.float32)
        }
