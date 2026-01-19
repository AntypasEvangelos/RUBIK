
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

from .losses import CorrespondenceLoss, RegularizationLoss
from ..models.frozen_energy import FrozenEnergyCanonicalizer

class CanonicalizerTrainer:
    def __init__(
        self,
        model: FrozenEnergyCanonicalizer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Only training the initialization head
        # The DINO backbone is frozen inside the model class
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=learning_rate
        )
        
        self.corr_loss_fn = CorrespondenceLoss()
        
    def train_epoch(self, epoch: int, wandb_run=None, log_interval: int = 50):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        total_loss = 0.0
        
        from .vis import visualize_batch
        
        for i, batch in enumerate(pbar):
            # Assume batch is a dict
            source_img = batch['image0'].to(self.device)
            target_img = batch['image1'].to(self.device)
            kpts0 = batch['keypoints0'].to(self.device) # Source pts
            kpts1 = batch['keypoints1'].to(self.device) # Target pts
            
            self.optimizer.zero_grad()
            
            # Forward pass
            warped_source, g_star = self.model(source_img, target_img)
            
            # Compute loss
            loss = self.corr_loss_fn(g_star, kpts0, kpts1)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if wandb_run is not None:
                wandb_run.log({"batch_loss": loss.item()})
                
                if i % log_interval == 0:
                    # Log visualization
                    with torch.no_grad():
                        vis = visualize_batch(source_img, target_img, warped_source, kpts0, kpts1)
                        import wandb
                        wandb_run.log({
                            "warps": wandb.Image(vis['warps'], caption=f"Epoch {epoch} Step {i}")
                        })
            
        return total_loss / len(self.train_loader)
