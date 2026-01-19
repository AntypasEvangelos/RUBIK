
import os
import argparse
import torch
from torch.utils.data import DataLoader

from canonicalizer.models.frozen_energy import FrozenEnergyCanonicalizer
from canonicalizer.training.trainer import CanonicalizerTrainer
from canonicalizer.training.datasets import RubikDataset

def main():
    parser = argparse.ArgumentParser(description="Train Canoncalizer on RUBIK")
    parser.add_argument("--data_path", type=str, default="rubik.json", help="Path to rubik.json")
    parser.add_argument("--nuscenes_path", type=str, default="/vast/projects/kostas/geometric-learning/nuscenes/nuscenes-download")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--wandb_project", type=str, default="rubik-canonicalizer", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    args = parser.parse_args()
    
    print(f"Training on device: {args.device}")
    
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args)
            )
            print("WandB initialized")
        except ImportError:
            print("WandB not installed, skipping logging")
    
    # 1. Dataset
    print("Loading dataset...")
    train_dataset = RubikDataset(
        data_path=args.data_path,
        nuscenes_path=args.nuscenes_path,
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    
    # 2. Model
    print(f"Initializing model {args.dino_model}...")
    model = FrozenEnergyCanonicalizer(
        dino_model=args.dino_model,
        num_iterations=5,
        step_size=0.1
    )
    
    # 3. Trainer
    trainer = CanonicalizerTrainer(
        model=model,
        train_loader=train_loader,
        learning_rate=args.lr,
        device=args.device
    )
    
    # 4. Train
    print("Starting training...")
    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(epoch, wandb_run=wandb_run)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        
        if wandb_run:
            wandb_run.log({"epoch_loss": avg_loss})
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/canonicalizer_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()
