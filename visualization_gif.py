import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import extract_xai_red_dots

# ==========================================
# 1. LOAD THE GLASS BOX
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("Loading Architecture...")
gcn = Spatial_GCN_Layer(in_channels=9, out_channels=64).to(device)
transformer = Temporal_Brain_Layer(embed_dim=64, num_heads=4, max_frames=100).to(device)

gcn.load_state_dict(torch.load('saved_weights/best_gcn.pth', map_location=device, weights_only=True))
transformer.load_state_dict(torch.load('saved_weights/best_transformer.pth', map_location=device, weights_only=True))
gcn.eval()
transformer.eval()

# ==========================================
# 2. RUN INFERENCE & EXTRACT HEATMAP
# ==========================================
print("Running Inference...")
dataset = NTUSkeletonDataset(data_folder='data/test_skeletons', max_frames=100)

# Grab the first file's engineered tensor for the neural network
engineered_tensor, true_label = dataset[0]

tensor_input = engineered_tensor.unsqueeze(0).to(device)
global_node = transformer.global_node

with torch.no_grad():
    B, T, M, V, C = tensor_input.shape
    gcn_input = tensor_input.permute(0, 2, 1, 3, 4).reshape(-1, T, V, C)
    gcn_features = gcn(gcn_input)
    transformer_input = torch.cat([gcn_features, global_node.expand(B*M, T, 1, 64)], dim=2)
    _, attention_matrix = transformer(transformer_input, return_attention=True)

raw_attention = attention_matrix[0].cpu().numpy()
heat_scores_100 = extract_xai_red_dots(raw_attention) # Shape: (100, 25)

# ==========================================
# 3. LOAD THE RAW SKELETON FOR PLOTTING
# ==========================================
# Get the filename (e.g., "S001C001...A010.pt") and find its original text file
pt_filename = dataset.file_list[0]
skeleton_filename = pt_filename.replace('.pt', '.skeleton')

# Build the path to the raw_text folder
raw_file_path = os.path.join('data/test_skeletons/raw_text', skeleton_filename)

# Use the dataset's built-in parser to get the un-padded, naturally moving data!
raw_skeleton_tensor = dataset.parse_single_skeleton(raw_file_path)
actual_frames = raw_skeleton_tensor.shape[0]

# Align the 100-frame AI attention to the raw video length
if actual_frames > 100:
    # If the video is longer (e.g., 118), stretch the 100 AI scores to fit perfectly
    indices = np.linspace(0, 99, actual_frames).astype(int)
    heat_scores = heat_scores_100[indices]
else:
    # If the video is shorter (e.g., 70), slice off the AI's zero-padded attention
    heat_scores = heat_scores_100[:actual_frames]

# Extract Person 0, XYZ coordinates
person_coords = raw_skeleton_tensor[:, 0, :, 0:3]

# ==========================================
# 4. BUILD THE GIF (Using Dynamic Camera Bounds)
# ==========================================
print(f"Generating XAI GIF Viewer for {actual_frames} frames...")

# Find the person's average physical location in the room
active_data = person_coords[person_coords[:, 0, 0] != 0] 
if len(active_data) > 0:
    mid_x = np.mean(active_data[:, :, 0])
    mid_y = np.mean(active_data[:, :, 1]) # Kinect Y is Height
    mid_z = np.mean(active_data[:, :, 2]) # Kinect Z is Depth
else:
    mid_x, mid_y, mid_z = 0, 0, 3 # Fallback

kinect_bones = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), 
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), 
    (22, 7), (23, 24), (24, 11)
]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('coolwarm') 

def update(frame_idx):
    ax.clear()
    
    # NEW: Dynamic Camera Boundaries!
    ax.set_xlim(mid_x - 1, mid_x + 1)
    ax.set_ylim(mid_z - 1, mid_z + 1) 
    ax.set_zlim(mid_y - 1, mid_y + 1)
    
    ax.set_title(f'Glass Box XAI | True Action: {true_label.item()} | Frame {frame_idx}')
    ax.set_xlabel('X (Meters)')
    ax.set_ylabel('Depth (Z Meters)')
    ax.set_zlabel('Height (Y Meters)')
    
    current_skeleton = person_coords[frame_idx]
    current_heat = heat_scores[frame_idx]
    
    # Skip if zero-padded (Though we sliced it, this is a good safety net)
    if np.all(current_skeleton[0] == 0):
        return
        
    xs, ys, zs = current_skeleton[:, 0], current_skeleton[:, 1], current_skeleton[:, 2]
    
    # Apply XAI Colors to Joints
    colors = cmap(current_heat)
    ax.scatter(xs, zs, ys, c=colors, s=60, edgecolors='black', linewidth=1, zorder=2)
    
    # Draw Bones
    for bone in kinect_bones:
        j1, j2 = bone
        ax.plot([xs[j1], xs[j2]], [zs[j1], zs[j2]], [ys[j1], ys[j2]], c='black', linewidth=2, zorder=1)

print(f"Compiling {actual_frames} frames into a GIF... Please wait.")
ani = animation.FuncAnimation(fig, update, frames=actual_frames, interval=50)

save_path = "xai_action.gif"
ani.save(save_path, writer='pillow', fps=20)
plt.close()
print(f"Success! Saved to {save_path}")