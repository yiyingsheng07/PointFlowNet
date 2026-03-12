from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import PointNet
from dataloader import PointDataset
import os
import random
import math
import numpy as np
from torch.utils.data import DataLoader
from loss import combined_loss, calculate_aneurysm_mae

def train():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 12
    num_epochs = 200
    model_choice = 4400
    pv_choice = 38000
    base_lr = 6e-4
    dataset_path = 'dataset/'
    train_annotation_path = 'train.txt'
    val_annotation_path = 'val.txt'
    norm_stats_path = os.path.join(dataset_path, "norm_stats_train.npz")


    # Create logs directory
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)

    # Load datasets
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = [line.strip() for line in f.readlines()]
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = [line.strip() for line in f.readlines()]
    
    train_dataset = PointDataset(
        filepath=dataset_path,
        filenames=train_lines,
        model_choice=model_choice,
        pv_choice=pv_choice,
        random_points=True,
        norm_stats_path=norm_stats_path
    )
    
    val_dataset = PointDataset(
        filepath=dataset_path,
        filenames=val_lines,
        model_choice=model_choice,
        pv_choice=pv_choice,
        random_points=True,
        norm_stats_path=norm_stats_path
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Initialize model
    model = PointNet().to(device)
    
    # Load pretrained weights if available
    pretrained_path = f'{logs_dir}/pretrained_weights.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
    #warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
    #main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6)
    
    # Training loop
    train_losses = []
    val_losses = []
    epoch_train_sac_maes = []  # Store average MAE for each epoch
    epoch_val_sac_maes = []    # Store average MAE for each epoch
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_sac_maes = []  # Collect MAE for each geometry
        
        for _, mx, pv, y, ori in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            mx, pv, y, ori = [t.to(device, non_blocking=True) for t in [mx, pv, y, ori]]
            
            optimizer.zero_grad()
            logits = model(mx, pv)
            loss = combined_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()

            # Calculate aneurysm MAE for each geometry in this batch
            batch_mae_values = calculate_aneurysm_mae(logits, y, ori)
            train_sac_maes.extend(batch_mae_values)  # Add all geometry MAEs to the list
        
        train_loss /= len(train_loader)

        # Calculate average MAE across all geometries
        avg_train_sac_mae = sum(train_sac_maes) / len(train_sac_maes) if train_sac_maes else 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_sac_maes = []  # Collect MAE for each geometry
        
        with torch.no_grad():
            for _, mx, pv, y, ori in tqdm(val_loader, desc=f'[Val]'):
                mx, pv, y, ori = [t.to(device, non_blocking=True) for t in [mx, pv, y, ori]]
                
                logits = model(mx, pv)
                loss = combined_loss(logits, y)
                val_loss += loss.item()

                # Calculate aneurysm MAE
                batch_mae_values = calculate_aneurysm_mae(logits, y, ori)
                val_sac_maes.extend(batch_mae_values)  # Add all geometry MAEs to the list

        val_loss /= len(val_loader)
        # Calculate average validation MAE
        avg_val_sac_mae = sum(val_sac_maes) / len(val_sac_maes) if val_sac_maes else 0.0
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_train_sac_maes.append(avg_train_sac_mae)
        epoch_val_sac_maes.append(avg_val_sac_mae)

        scheduler.step(val_loss)  # Update learning rate based on validation loss
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{logs_dir}/best_model.pth')
            print(f"Best model saved at epoch {epoch+1}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f'{logs_dir}/epoch_{epoch+1}_model.pth')

        # Log progress
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}")
        print(f"            Train Sac MAE = {avg_train_sac_mae:.8f}, Val Sac MAE = {avg_val_sac_mae:.8f}")

        with open(f"{logs_dir}/losses.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss = {train_loss:.8f}, Val Loss = {val_loss:.8f}, "
                   f"Train Sac MAE = {avg_train_sac_mae:.8f}, Val Sac MAE = {avg_val_sac_mae:.8f}\n")
    
    # Plot final results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{logs_dir}/loss_plot.png')
    #plt.show()

    # Plot aneurysm sac MAE
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_train_sac_maes, label='Train Sac MAE')
    plt.plot(range(1, num_epochs + 1), epoch_val_sac_maes, label='Val Sac MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Aneurysm Sac MAE')
    plt.title('Aneurysm Sac MAE Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{logs_dir}/sac_mae_plot.png')

if __name__ == '__main__':
    train()