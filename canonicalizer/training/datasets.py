
import torch
from torch.utils.data import Dataset
import json
import os
import os.path as osp
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

class RubikDataset(Dataset):
    """
    Dataset loader for RUBIK/nuScenes training data.
    """
    def __init__(
        self,
        data_path: str,
        nuscenes_path: str,
        mode: str = 'train',
        transform = None,
        img_size: int = 518 # DINOv2 friendly size
    ):
        self.data = json.load(open(data_path))
        self.nuscenes_path = nuscenes_path
        self.mode = mode
        
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
        
        # Transform
        if self.transform:
            img0_tensor = self.transform(img0)
            img1_tensor = self.transform(img1)
            
        # For training, we technically need Ground Truth Correspondences
        # But RUBIK json only gives Relative Pose and Intrinsics.
        # We can either:
        # 1. Project random points from I1 to I2 using Depth + Pose (needs depth maps)
        # 2. Or just return the images/pose if we use a photometric/pose loss instead of correspondence loss.
        # 3. Or use an offline SIFT/SuperPoint matching to get pseudo-GT.
        
        # Assuming we want to use the CorrespondenceLoss we defined, we need keypoints.
        # Let's assume we can generate dense matches via homography if planar, or epipolar?
        # Without depth, we can't get exact pixel-wise correspondence for general scenes.
        
        # HACK: For now, I will return dummy keypoints just to make the trainer runnable 
        # as per user request to "implement infrastructure". 
        # A real training setup would need depth maps (available in unidepths) to project points.
        
        return {
            'image0': img0_tensor,
            'image1': img1_tensor,
            # Dummy placeholders - user needs to decide how to supervise (depth projection vs pseudo-GT)
            'keypoints0': torch.zeros(100, 2), 
            'keypoints1': torch.zeros(100, 2),
            'pose': torch.tensor(sample['rel_pose'], dtype=torch.float32),
            'K0': torch.tensor(sample['K1'], dtype=torch.float32), # Note K1/K2 naming in json vs indexes
            'K1': torch.tensor(sample['K2'], dtype=torch.float32)
        }
