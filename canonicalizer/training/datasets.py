
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
            
        # For dense matching methods (RoMa, LoFTR, etc.), we don't need sparse keypoints.
        # Instead, we rely on:
        # 1. Photometric loss (image similarity) in feature space
        # 2. Pose supervision using ground truth relative pose
        # 3. The frozen DINO energy landscape guides canonicalization
        #
        # This approach is more suitable for dense matchers than sparse correspondences.

        return {
            'image0': img0_tensor,
            'image1': img1_tensor,
            'pose': torch.tensor(sample['rel_pose'], dtype=torch.float32),
            'K0': torch.tensor(sample['K1'], dtype=torch.float32),
            'K1': torch.tensor(sample['K2'], dtype=torch.float32),
            'original_size0': torch.tensor([H0, W0], dtype=torch.float32),
            'original_size1': torch.tensor([H1, W1], dtype=torch.float32)
        }
