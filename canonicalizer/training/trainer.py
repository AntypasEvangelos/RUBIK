
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

from .losses import PhotometricLoss, CorrespondenceLoss, RegularizationLoss
from ..models.frozen_energy import FrozenEnergyCanonicalizer

class CanonicalizerTrainer:
    def __init__(
        self,
        model: FrozenEnergyCanonicalizer,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        loss_type: str = 'photometric'  # 'photometric' for dense, 'correspondence' for sparse
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_type = loss_type

        # Only training the initialization head
        # The DINO backbone is frozen inside the model class
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )

        # Choose loss function based on whether we have keypoints or not
        if loss_type == 'photometric':
            self.loss_fn = PhotometricLoss(loss_type='cosine')
        else:
            self.loss_fn = CorrespondenceLoss()
        
    def train_epoch(self, epoch: int, wandb_run=None, log_interval: int = 50):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        total_loss = 0.0
        
        from .vis import visualize_batch

        for i, batch in enumerate(pbar):
            # Assume batch is a dict
            source_img = batch['image0'].to(self.device)
            target_img = batch['image1'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            warped_source, g_star = self.model(source_img, target_img)

            # Compute loss based on loss type
            if self.loss_type == 'photometric':
                # For photometric loss, we need features
                # The model already computed features internally, but we need them here
                # Extract features from warped source and target
                with torch.no_grad():
                    target_feats = self.model.dino(target_img)[self.model.feature_layer]

                # Warped source features (with gradient)
                warped_feats = self.model.dino(warped_source)[self.model.feature_layer]

                # Compute photometric loss in feature space
                loss = self.loss_fn(warped_feats, target_feats)
            else:
                # Correspondence loss (if keypoints are provided)
                kpts0 = batch['keypoints0'].to(self.device)
                kpts1 = batch['keypoints1'].to(self.device)
                loss = self.loss_fn(g_star, kpts0, kpts1)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

            if wandb_run is not None:
                wandb_run.log({"batch_loss": loss.item()})

                if i % log_interval == 0:
                    # Log visualization
                    with torch.no_grad():
                        kpts0 = batch.get('keypoints0', None)
                        kpts1 = batch.get('keypoints1', None)
                        if kpts0 is not None:
                            kpts0 = kpts0.to(self.device)
                            kpts1 = kpts1.to(self.device)
                        vis = visualize_batch(source_img, target_img, warped_source, kpts0, kpts1)
                        import wandb
                        wandb_run.log({
                            "warps": wandb.Image(vis['warps'], caption=f"Epoch {epoch} Step {i}")
                        })

        return total_loss / len(self.train_loader)
