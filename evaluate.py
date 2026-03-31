import os
import torch
import torch.nn as nn
import numpy as np
from  torch.utils.data import DataLoader
from tqdm import tqdm

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import extract_xai_red_dots

# 1. HARDWARE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in-use: {device.type.upper()}")

# 2. INITIALIZATION
VAL_DIR = 'data/val_skeletons'
BATCH_SIZE = 16
NUM_CLASSES = 60

print("Loading Data..")
val_dataset = NTUSkeletonDataset(data_folder=VAL_DIR,max_frames=100)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# 3. Model
gcn = Spatial_GCN_Layer().to(device)
transformer = Temporal_Brain_Layer().to(device)
classifier = nn.Linear(64, NUM_CLASSES).to(device)

gcn.load_state_dict(torch.load('saved_weights/gcn_epoch_50.pth', map_location=device, weights_only=True))
transformer.load_state_dict(torch.load('saved_weights/transformer_epoch_50.pth', map_location=device, weights_only=True))
classifier.load_state_dict(torch.load('saved_weights/classifier_epoch_50.pth', map_location=device, weights_only=True))
global_node = transformer.global_node  # nn.Parameter saved inside transformer's state_dict

gcn.eval()
transformer.eval()
classifier.eval()

# 4. Evaluation Loop
print("Running Evaluation..")

total_samples = 0
correct_predictions = 0
saved_xai = False

loop = tqdm(val_dataloader, total=len(val_dataloader), leave=True, desc="Evaluating")
with torch.no_grad():
    for batch_idx, (batched_data, labels) in enumerate(loop):
        batched_data = batched_data.to(device)
        labels = labels.to(device)

        B, T, M, V, C = batched_data.shape
        gcn_input = batched_data.permute(0, 2, 1, 3, 4).reshape(-1, T, V, C)

        # Forward Pass
        gcn_features = gcn(gcn_input)
        frames = gcn_features.shape[1]
        
        global_node_expanded = global_node.expand(B*M, frames, 1, 64)
        transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)

        attn_output, real_attention_matrix = transformer(transformer_input, return_attention=True)
        
        global_node_features = attn_output[:, :, 25, :] 
        final_video_features = torch.mean(global_node_features, dim=1)

        separated_bodies = final_video_features.view(B, M, 64)
        video_representation, _ = torch.max(separated_bodies, dim=1) 
        
        predictions = classifier(video_representation)
        
        # Calculate Accuracy
        _, predicted_classes = torch.max(predictions, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += labels.size(0)

        current_acc = (correct_predictions / total_samples)*100
        loop.set_postfix(acc=f"{current_acc:.2f}%")

        # --- XAI HEATMAP EXTRACTION (Only for the first video) ---
        if not saved_xai:
            print("Extracting Glass-Box XAI Heatmap for Sample 1...")
            xai_data = real_attention_matrix[0].cpu().numpy()
            heat_scores = extract_xai_red_dots(xai_data)
            np.save("extracted_xai_heatmap.npy", heat_scores)
            saved_xai = True

# 4. Final Report
accuracy = (correct_predictions / total_samples) * 100
print("-" * 50)
print("EVALUATION COMPLETE")
print(f"Total Unseen Videos Processed: {total_samples}")
print(f"Final Top-1 Accuracy:          {accuracy:.2f}%")
print("-> XAI Heatmap saved to root directory as 'extracted_xai_heatmap.npy'")
print("-" * 50)