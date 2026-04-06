import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import extract_xai_red_dots

# ==========================================
# 0. SETUP DIRECTORIES
# ==========================================
NPY_DIR = "results/xai_npy"
GIF_DIR = "results/xai_gifs"
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

# ==========================================
# 1. LOAD THE GLASS BOX
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Loading Architecture on {device}...")
gcn = Spatial_GCN_Layer(in_channels=9, out_channels=64).to(device)
transformer = Temporal_Brain_Layer(embed_dim=64, num_heads=4, max_frames=100).to(device)

gcn.load_state_dict(torch.load('saved_weights/best_gcn.pth', map_location=device, weights_only=True))
transformer.load_state_dict(torch.load('saved_weights/best_transformer.pth', map_location=device, weights_only=True))
gcn.eval()
transformer.eval()

# ==========================================
# 2. LOAD DATASET
# ==========================================
print("Loading Test Dataset...")
dataset = NTUSkeletonDataset(data_folder='data/test_skeletons', max_frames=100)
total_files = len(dataset.file_list)

kinect_bones = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), 
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), 
    (22, 7), (23, 24), (24, 11)
]

# ==========================================
# 3. BATCH PROCESSING LOOP
# ==========================================
print(f"Starting batch processing of {total_files} files...")

for file_idx in tqdm(range(total_files), desc="Processing XAI"):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    
    npy_path = os.path.join(NPY_DIR, f"{target_base}.npy")
    gif_path = os.path.join(GIF_DIR, f"{target_base}.gif")
    
    # SMART RESUME: Skip if both files already exist
    if os.path.exists(npy_path) and os.path.exists(gif_path):
        continue

    # --- A. RUN INFERENCE ---
    engineered_tensor, true_label = dataset[file_idx]
    tensor_input = engineered_tensor.unsqueeze(0).to(device)
    global_node = transformer.global_node

    with torch.no_grad():
        B, T, M, V, C = tensor_input.shape
        gcn_input = tensor_input.permute(0, 2, 1, 3, 4).reshape(-1, T, V, C)
        gcn_features = gcn(gcn_input)
        transformer_input = torch.cat([gcn_features, global_node.expand(B*M, T, 1, 64)], dim=2)
        _, attention_matrix = transformer(transformer_input, return_attention=True)

    raw_attention = attention_matrix[0].cpu().numpy()
    heat_scores_100 = extract_xai_red_dots(raw_attention)

    # Save NPY
    if not os.path.exists(npy_path):
        np.save(npy_path, heat_scores_100)

    # --- B. RENDER GIF ---
    if not os.path.exists(gif_path):
        skeleton_filename = target_base + '.skeleton'
        raw_file_path = os.path.join('data/test_skeletons/raw_text', skeleton_filename)
        
        raw_skeleton_tensor = dataset.parse_single_skeleton(raw_file_path)
        
        # Failsafe if the raw text file is missing or corrupted
        if raw_skeleton_tensor is None:
            continue
            
        actual_frames = raw_skeleton_tensor.shape[0]
        
        # 1. SMART TIME-WARPING (Fixes the >100 frames crash)
        if actual_frames > 100:
            indices = np.linspace(0, 99, actual_frames).astype(int)
            heat_scores = heat_scores_100[indices]
        else:
            heat_scores = heat_scores_100[:actual_frames]
            
        person_coords = raw_skeleton_tensor[:, 0, :, 0:3]

        # 2. DYNAMIC CAMERA BOUNDS (Fixes the blank screen)
        # Find the person's average physical location in the room
        active_data = person_coords[person_coords[:, 0, 0] != 0] 
        if len(active_data) == 0:
            continue # Failsafe if the body was completely empty
            
        mid_x = np.mean(active_data[:, :, 0])
        mid_y = np.mean(active_data[:, :, 1]) # Kinect Y is Height
        mid_z = np.mean(active_data[:, :, 2]) # Kinect Z is Depth

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('coolwarm') 

        def update(frame_idx):
            ax.clear()
            
            # Lock the camera to a perfect 2-meter cube centered exactly on the actor!
            ax.set_xlim(mid_x - 1, mid_x + 1)
            ax.set_ylim(mid_z - 1, mid_z + 1) # Matplotlib Y axis handles Depth
            ax.set_zlim(mid_y - 1, mid_y + 1) # Matplotlib Z axis handles Height
            
            ax.set_title(f'Action: {true_label.item()} | File: {target_base}')
            ax.set_xlabel('X (Meters)')
            ax.set_ylabel('Depth (Z Meters)')
            ax.set_zlabel('Height (Y Meters)')
            
            current_skeleton = person_coords[frame_idx]
            current_heat = heat_scores[frame_idx]
            
            if np.all(current_skeleton[0] == 0):
                return
                
            xs, ys, zs = current_skeleton[:, 0], current_skeleton[:, 1], current_skeleton[:, 2]
            colors = cmap(current_heat)
            ax.scatter(xs, zs, ys, c=colors, s=60, edgecolors='black', linewidth=1, zorder=2)
            
            for bone in kinect_bones:
                j1, j2 = bone
                ax.plot([xs[j1], xs[j2]], [zs[j1], zs[j2]], [ys[j1], ys[j2]], c='black', linewidth=2, zorder=1)

        # Disable printing inside the loop so it doesn't break the progress bar
        ani = animation.FuncAnimation(fig, update, frames=actual_frames, interval=50)
        ani.save(gif_path, writer='pillow', fps=20)
        
        # CRITICAL: Close the figure to prevent RAM memory leaks!
        plt.close(fig) 

print("Batch Processing Complete! Check the 'results/xai_npy' and 'results/xai_gifs' folders.")