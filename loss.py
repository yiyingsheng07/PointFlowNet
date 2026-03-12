import torch
import torch.nn as nn
import math


def combined_loss(pred, target):
    # vector-component MAE (same scaling as combined_loss)
    vec_mae = (pred - target).abs().sum(dim=-1)     # B,N  (sum, not mean!)

    # magnitude MAE
    mag_mae = (pred.norm(dim=-1) - target.norm(dim=-1)).abs()  # B,N

    return (vec_mae + mag_mae).mean()


def calculate_aneurysm_mae(logits, y, ori_xyz):
    """
    Calculate Mean Absolute Error in the aneurysm sac region for each geometry in the batch.
    
    Args:
        logits: Model predictions [batch_size, num_points, 1]
        y: Ground truth labels [batch_size, num_points, 1] 
        ori_xyz: Original xyz coordinates [batch_size, num_points, 3]
    
    Returns:
        mae_values: List of MAE values for each geometry in the batch
    """
    # Define aneurysm sac region - you may need to adjust the y upper bound
    sac_mask = (
        (ori_xyz[..., 0] >= -0.003) & 
        (ori_xyz[..., 0] <= 0.003) & 
        (ori_xyz[..., 1] >= -0.0033) & 
        (ori_xyz[..., 1] <= 0.0033)  # Added upper bound - adjust as needed
    ).float()
    
    # Expand mask to match logits/y dimensions if needed
    if sac_mask.dim() == 2:  # [batch_size, num_points]
        sac_mask = sac_mask.unsqueeze(-1)  # [batch_size, num_points, 1]
    
    batch_size = logits.shape[0]
    mae_values = []
    
    # Calculate MAE for each geometry individually
    for i in range(batch_size):
        geometry_sac_mask = sac_mask[i]  # [num_points, 1]
        geometry_logits = logits[i]      # [num_points, 1]
        geometry_y = y[i]                # [num_points, 1]
        
        # Get sac points for this geometry
        sac_points_mask = geometry_sac_mask.squeeze(-1) > 0  # [num_points]
        
        if torch.sum(sac_points_mask) > 0:  # If geometry has sac points
            sac_logits = geometry_logits[sac_points_mask]
            sac_y = geometry_y[sac_points_mask]
            
            # Calculate MAE for this geometry's sac region
            mae = torch.mean(torch.abs(sac_logits - sac_y))
            mae_values.append(mae.item())
        else:
            # Geometry has no sac points
            mae_values.append(0.0)
    
    return mae_values