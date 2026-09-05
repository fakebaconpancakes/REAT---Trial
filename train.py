import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset

# =====================
# 1. HYPERPARAMETERS
# =====================
DATA_DIR = 'data/xsub/train_skeletons' #CHANGE THIS DIRECTORY!!!
VAL_DIR = 'data/xsub/val_skeletons'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 60

if __name__ == '__main__':
    wandb.init(
        project="HAR-REAT",
        name="Run13-128-Decoupled-Bodies",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": "Spatial GCN + Transformer",
            "dataset": "NTU-RGB+D",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "scheduler": "10-Epoch Linear Warmup + Cosine Decay"
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Device Type: {device.type.upper()}')

    # =====================
    # 2. INTIALIZATION
    # =====================
    print("Loading Dataset..")
    dataset = NTUSkeletonDataset(data_folder=DATA_DIR, max_frames=100, is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    print("Loading Validation Dataset..")
    val_dataset = NTUSkeletonDataset(data_folder=VAL_DIR, max_frames=100, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # The Neural Networks
    gcn = Spatial_GCN_Layer(in_channels=9, out_channels=128).to(device)
    transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
    global_node = transformer.global_node

    # The Classifier
    classifier = nn.Linear(128, NUM_CLASSES).to(device)

    wandb.watch(gcn, log="all", log_freq=10)
    wandb.watch(transformer, log="all", log_freq=10)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(gcn.parameters()) + list(transformer.parameters()) + list(classifier.parameters()),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )

    # --- UPGRADE: Linear Warmup + Cosine Annealing ---
    warmup_epochs = 10
    cosine_epochs = EPOCHS - warmup_epochs

    # 1. Warmup: Start at 1% of the LR, ramp up to 100% over 10 epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    # 2. Cosine Decay: Take over at Epoch 11, curve down to near-zero by Epoch 100
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)

    # 3. Stitch them together
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # =====================
    # 3. TRAINING LOOP
    # =====================
    print("Start Training..")

    gcn.train()
    transformer.train()
    classifier.train()
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        loop = tqdm(dataloader, total=len(dataloader), leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for batch_idx, (batched_data, body_mask, labels) in enumerate(loop):
            batched_data = batched_data.to(device)
            body_mask = body_mask.to(device)
            labels = labels.to(device)

            B, M, T, V, C = batched_data.shape
            gcn_input = batched_data.reshape(B*M, T, V, C)# (Batch*Bodies, Time, Joints, Channels)

            # 1. Forward Pass (GCN)
            gcn_features = gcn(gcn_input) # Output: (Batch*2, Time, Joints, 64)

            # 2. Attach Global Node
            frames = gcn_features.shape[1]
            global_node_expanded = global_node.expand(B*M, frames, 1, 128)
            transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)

            # 3. Forward Pass (Transformer)
            video_representation = transformer(transformer_input, B, M, body_mask=body_mask)

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

        # =====================
        # 4. VALIDATION LOOP
        # =====================
        gcn.eval()
        transformer.eval()
        classifier.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad(): # Turns off gradient tracking to save memory/speed
            for val_batch, val_mask, val_labels in val_dataloader:
                val_batch = val_batch.to(device)
                val_mask = val_mask.to(device)
                val_labels = val_labels.to(device)

                
                v_B, v_M, v_T, v_V, v_C = val_batch.shape

                # Validation Folding Trick
                val_gcn_input = val_batch.reshape(v_B*v_M, v_T, v_V, v_C) # (Batch*Bodies, Time, Joints, Channels)
                v_gcn_feat = gcn(val_gcn_input)
                
                v_frames = v_gcn_feat.shape[1]
                v_global_node = global_node.expand(v_B*v_M, v_frames, 1, 128)
                v_transformer_input = torch.cat([v_gcn_feat, v_global_node], dim=2)

                # Validation Intercation Call
                v_vid_rep = transformer(v_transformer_input, v_B, v_M, body_mask=val_mask)
                
                v_preds = classifier(v_vid_rep)
                v_loss = criterion(v_preds, val_labels)
                
                val_loss += v_loss.item()
                _, v_pred_classes = torch.max(v_preds, 1)
                val_correct += (v_pred_classes == val_labels).sum().item()
                val_samples += val_labels.size(0)

        epoch_val_loss = val_loss / len(val_dataloader)
        epoch_val_acc = (val_correct / val_samples) * 100

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs('saved_weights', exist_ok=True)
            torch.save(gcn.state_dict(), 'saved_weights/best_gcn.pth')
            torch.save(transformer.state_dict(), 'saved_weights/best_transformer.pth')
            torch.save(classifier.state_dict(), 'saved_weights/best_classifier.pth')
            print(f"🌟 New Best Model! Saved with Val Acc: {best_val_acc:.2f}%")

        # Put models back into training mode for the next epoch!
        gcn.train()
        transformer.train()
        classifier.train()

        # Step the Scheduler at the end of the Epoch
        scheduler.step()
        
        # Extract current Learning Rate to track it
        current_lr = optimizer.param_groups[0]['lr']

        # Print Epoch Report
        epoch_accuracy = (correct_predictions / total_samples) * 100
        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_accuracy,
            "Validation Loss": epoch_val_loss,
            "Validation Accuracy": epoch_val_acc,
            "Learning Rate": current_lr
        })

        # Save weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs('saved_weights', exist_ok=True)
            torch.save(gcn.state_dict(), f'saved_weights/gcn_epoch_{epoch+1}.pth')
            torch.save(transformer.state_dict(), f'saved_weights/transformer_epoch_{epoch+1}.pth')
            torch.save(classifier.state_dict(), f'saved_weights/classifier_epoch_{epoch+1}.pth')
            print(f"-> Checkpoint saved for Epoch {epoch+1}")

    print("Training Complete!")


