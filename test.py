import torch
import torch.nn as nn
import csv
import time
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import PointNet
from dataloader import PointDataset


# Configuration Parameters
batch_size = 12
model_choice = 4400
pv_choice = 38000

# Folder configurations
logs_dir = 'logs'
results_dir = os.path.join(logs_dir, 'test_results')

dataset_path = 'dataset/'
test_annotation_path = 'test.txt'
norm_stats_path = os.path.join(dataset_path, "norm_stats_train.npz")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data filenames
with open(test_annotation_path, encoding='utf-8') as f:
    test_lines = [line.strip() for line in f.readlines()]

# Initialize the test dataset and data loader
test_dataset = PointDataset(
    filepath=dataset_path,
    filenames=test_lines,
    model_choice=model_choice,
    pv_choice=pv_choice,
    random_points=False,
    norm_stats_path=norm_stats_path
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8, 
    pin_memory=True
)

# Load trained model
model = PointNet().to(device)


model.load_state_dict(torch.load(f'{logs_dir}/best_model.pth', map_location=device, weights_only=True))
model.eval()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}")


# Function to calculate the magnitude
def calculate_magnitude(tensor):
    return torch.sqrt(torch.sum(tensor**2, dim=2))

# Function to evaluate the model on the test data
def evaluate_model(loader):
    start_time = time.time()
    with torch.no_grad():
        loss_results = {}
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            filenames, mx, pv, y, original_pv = batch
            mx = mx.to(device, non_blocking=True)
            pv = pv.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Keep tensors as PyTorch tensors instead of converting to numpy
            original_pv_cpu = original_pv.cpu()

            logits = model(mx, pv)
            logits_cpu = logits.cpu()
            y_cpu = y.cpu()
            pv_cpu = pv.cpu()

            pred_magnitudes = calculate_magnitude(logits)
            true_magnitudes = calculate_magnitude(y)
            pred_magnitudes_cpu = pred_magnitudes.cpu()
            true_magnitudes_cpu = true_magnitudes.cpu()

            for filename, pred_magnitude, true_magnitude in zip(filenames, pred_magnitudes_cpu, true_magnitudes_cpu):
                loss = nn.L1Loss()(pred_magnitude, true_magnitude).item()
                loss_results[filename] = loss

            # Save predictions and ground truth to CSV files, one per filename
            if not os.path.exists(results_dir):
                os.mkdir(results_dir)
                
            for filename, preds, truths, pvs, pred_magnitude, true_magnitude in zip(
                    filenames, logits_cpu, y_cpu, original_pv_cpu, 
                    pred_magnitudes_cpu, true_magnitudes_cpu):
                output_filename = f"{results_dir}/{filename}_results.csv"
                with open(output_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['x', 'y', 'z', 'truth_velocity u', 'truth_velocity v', 'truth_velocity w',
                                     'predict_velocity u', 'predict_velocity v', 'predict_velocity w',
                                     'pred_magnitude', 'truth_magnitude'])
                    
                    # Convert tensors to lists directly without using numpy
                    for i in range(pvs.size(0)):
                        pv_point = [pvs[i][0].item(), pvs[i][1].item(), pvs[i][2].item()]
                        truth = [truths[i][0].item(), truths[i][1].item(), truths[i][2].item()]
                        pred = [preds[i][0].item(), preds[i][1].item(), preds[i][2].item()]
                        pmag = pred_magnitude[i].item()
                        tmag = true_magnitude[i].item()
                        writer.writerow([*pv_point, *truth, *pred, pmag, tmag])
        
        # Save the loss results to a text file
        with open(f"{results_dir}/magnitude_loss_results.txt", "w") as f:
            for filename, loss in loss_results.items():
                f.write(f"{filename}: {loss:.8f}\n")

        end_time = time.time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        
# Evaluate the model
if __name__ == '__main__':
    evaluate_model(test_loader)