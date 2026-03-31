import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset

# =====================
# 1. HYPERPARAMETERS
# =====================
DATA_DIR = 'data/train_skeletons' #CHANGE THIS DIRECTORY!!!
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 60

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f'Device Type: {device.type.upper()}')

# =====================
# 2. INTIALIZATION
# =====================
print("Loading Dataset..")
dataset = NTUSkeletonDataset(data_folder=DATA_DIR, max_frames=100)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# The Neural Networks
gcn = Spatial_GCN_Layer().to(device)
transformer = Temporal_Brain_Layer().to(device)
global_node = nn.Parameter(torch.randn(1, 1, 1, 64, device=device))

# The Classifier
classifier = nn.Linear(64, NUM_CLASSES).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(gcn.parameters()) + list(transformer.parameters()) + [global_node] + list(classifier.parameters()),
    lr = LEARNING_RATE
)

# =====================
# 3. TRAINING LOOP
# =====================
print("Start Training..")

gcn.train()
transformer.train()
classifier.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    loop = tqdm(dataloader, total=len(dataloader), leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for batch_idx, (batched_data,labels) in enumerate(loop):
        # batched_data shape: (Batch, Time=100, Bodies=2, Joints=25, Channels=3)
        # Note: We need fake labels for this script to run. In your real dataset, 
        # you will extract the actual label from the filename (e.g., A060 = class 59)
        batched_data = batched_data.to(device)
        labels = labels.to(device)

        B, T, M, V, C = batched_data.shape
        gcn_input = batched_data.permute(0, 2, 1, 3, 4).reshape(-1, T, V, C).to(device)

        # 1. Forward Pass (GCN)
        gcn_features = gcn(gcn_input) # Output: (Batch*2, Time, Joints, 64)

        # 2. Attach Global Node
        frames = gcn_features.shape[1]
        global_node_expanded = global_node.expand(B*M, frames, 1, 64)
        transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)

        # 3. Forward Pass (Transformer)
        attn_output = transformer(transformer_input) # Output: (Batch*2, 64)

        # 4. Global Node (index 25) thought at the VERY LAST frame (index -1)
        final_video_features = attn_output[:, -1, 25, :] # Output: (Batch*2, 64)

        separated_bodies = final_video_features.view(B, M, 64)
        # Compress them by taking the strongest signal between Person 1 and Person 2
        video_representation, _ = torch.max(separated_bodies, dim=1) # Output: (Batch, 64)

        # 5. Predictions
        predictions = classifier(video_representation)

        # 6. Loss
        loss = criterion(predictions, labels)

        # 7. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted_classes = torch.max(predictions, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    # Print Epoch Report
    epoch_accuracy = (correct_predictions / total_samples) * 100
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_accuracy:.2f}%")
    
    # Save weights every 10 epochs
    if (epoch + 1) % 10 == 0:
        os.makedirs('saved_weights', exist_ok=True)
        torch.save(gcn.state_dict(), f'saved_weights/gcn_epoch_{epoch+1}.pth')
        torch.save(transformer.state_dict(), f'saved_weights/transformer_epoch_{epoch+1}.pth')
        torch.save(classifier.state_dict(), f'saved_weights/classifier_epoch_{epoch+1}.pth')
        print(f"-> Checkpoint saved for Epoch {epoch+1}")

print("Training Complete!")


