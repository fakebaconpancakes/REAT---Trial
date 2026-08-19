import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

import argparse
import torch.nn as nn

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import extract_xai_red_dots
from train import NUM_CLASSES

NTU_CLASSES = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup',
    'throw', 'sitting down', 'standing up', 'clapping', 'reading', 'writing',
    'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe',
    'wear on glasses', 'take off glasses', 'put on a hat/cap', 'take off a hat/cap',
    'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping',
    'jump up', 'make a phone call', 'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
    'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute',
    'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough', 'staggering',
    'falling', 'touch head (headache)', 'touch chest (stomachache)', 'touch back (backache)', 'touch neck (neckache)',
    'nausea or vomiting condition', 'use a fan/feeling warm', 'punching/slapping other person',
    'kicking other person', 'pushing other person', 'pat on back of other person',
    'point finger at the other person', 'hugging other person', 'giving something to other person',
    'touch other person\'s pocket', 'handshaking', 'walking towards each other', 'walking apart from each other'
]

# ==========================================
# 0. SETUP DIRECTORIES
# ==========================================
NPY_DIR = "results/xai_npy"
RAW_NPY_DIR = "results/xai_raw_npy"
DIFF_NPY_DIR = "results/xai_diff_npy"
GIF_DIR = "results/xai_gifs"
FRAME_DIR = 'results/xai_frames'

os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(RAW_NPY_DIR, exist_ok=True)
os.makedirs(DIFF_NPY_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

# ==========================================
# 1. LOAD THE GLASS BOX
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Loading Architecture on {device}...")
gcn = Spatial_GCN_Layer(in_channels=9, out_channels=128).to(device)
transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
classifier = nn.Linear(128, NUM_CLASSES).to(device)

gcn.load_state_dict(torch.load('saved_weights/best_gcn.pth', map_location=device, weights_only=True))
transformer.load_state_dict(torch.load('saved_weights/best_transformer.pth', map_location=device, weights_only=True))
classifier.load_state_dict(torch.load('saved_weights/best_classifier.pth', map_location=device, weights_only=True))
gcn.eval()
transformer.eval()
classifier.eval()

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

parser = argparse.ArgumentParser(description='Batch process XAI heatmaps.')
parser.add_argument('--start', type=int, default=0, help='Start index for processing files.')
parser.add_argument('--end', type=int, default=None, help='End index for processing files.')
args = parser.parse_args()

start_idx = args.start
end_idx = args.end if args.end is not None else total_files

print(f"Starting batch processing from file index {start_idx} to {end_idx}...")

for file_idx in tqdm(range(start_idx, end_idx), desc="Processing XAI"):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    
    npy_path = os.path.join(NPY_DIR, f"{target_base}.npy")
    raw_npy_path = os.path.join(RAW_NPY_DIR, f"{target_base}.npy")
    diff_npy_path = os.path.join(DIFF_NPY_DIR, f"{target_base}.npy")
    gif_path = os.path.join(GIF_DIR, f"{target_base}.gif")
    
    # SMART RESUME: Skip if both files already exist
    if os.path.exists(npy_path) and os.path.exists(raw_npy_path) and os.path.exists(diff_npy_path) and os.path.exists(gif_path):
        continue

    # --- A. RUN INFERENCE ---
    engineered_tensor, true_label = dataset[file_idx]
    tensor_input = engineered_tensor.unsqueeze(0).to(device)
    global_node = transformer.global_node

    with torch.no_grad():
        B, M, T, V, C = tensor_input.shape
        gcn_input = tensor_input.reshape(B * M, T, V, C)
        gcn_features = gcn(gcn_input)

        frames = gcn_features.shape[1]
        transformer_input = torch.cat([gcn_features, global_node.expand(B * M, frames, 1, 128)], dim=2)

        video_representation, attention_matrix = transformer(
            transformer_input, B, M, body_mask=body_mask, return_attention=True
        )
        
        # ==========================================
        # DIAGNOSTIC PRINT: EXPOSING THE AI'S BRAIN
        # ==========================================
        raw_probs = attention_matrix[0].cpu().numpy()
        
        # 1. Find max attention per frame (collapsed across bodies and joints)
        frame_max = np.max(raw_probs, axis=(0, 2)) # Shape: (T,)
        
        # 2. Find the Raw Bias Peak Frame and Resting Frame
        raw_peak_frame = np.argmax(frame_max)
        resting_frame = np.argmin(frame_max)

        # Get the resting attention specifically for both bodies at that frame -> Shape: (M, 25)
        resting_attention = raw_probs[:, resting_frame, :]
        resting_attention_expanded = resting_attention[:, np.newaxis, :] # Shape: (M, 1, 25)
        
        # 3. RELATIVE DIFFERENTIAL XAI (Fold Change Normalization)
        # We add epsilon to prevent dividing by absolute zero
        epsilon = 1e-6
        differential_matrix = np.maximum((raw_probs - resting_attention_expanded) / (resting_attention_expanded + epsilon), 0)

        # Get exact coordinates (Body, Frame, Joint) of the peak shift
        diff_peak_body, diff_peak_frame, true_max_joint = np.unravel_index(np.argmax(differential_matrix), differential_matrix.shape)
        
        # Get the attention profile for that specific body at that specific peak frame
        dynamic_attention = differential_matrix[diff_peak_body, diff_peak_frame]
        
        print("\n" + "="*40)
        print(f"RAW BIAS PEAK FRAME: {raw_peak_frame}")
        print(f"DIFFERENTIAL PEAK FRAME: {diff_peak_frame}")
        print(f"DIFFERENTIAL PEAK BODY: Person {diff_peak_body + 1}")
        print(f"ACTIVITY: {NTU_CLASSES[true_label.item()]}")
        print("="*40)
        print(f"TRUE #1 DYNAMIC JOINT: Joint {true_max_joint} (+{dynamic_attention[true_max_joint]:.6f} shift)")
        print("-" * 40)
        print(f"Head (Joint 3) Shift:             +{dynamic_attention[3]:.6f}")
        print(f"Right Hand (Joint 24) Shift:      +{dynamic_attention[24]:.6f}")
        print(f"Left Foot (Joint 15) Shift:       +{dynamic_attention[15]:.6f}")
        print(f"Spine Base (Joint 0) Shift:       +{dynamic_attention[0]:.6f}")
        print("="*40 + "\n")
        # ==========================================

        predictions = classifier(video_representation)
        probs = torch.nn.functional.softmax(predictions, dim=1)
        pred_prob, predicted_class = torch.max(probs, 1)
        pred_label = predicted_class.item()
        pred_confidence = pred_prob.item() * 100

    # Save NPY maps
    raw_heat_scores_100 = extract_xai_red_dots(raw_probs)
    diff_heat_scores_100 = extract_xai_red_dots(differential_matrix)

    # Save NPY
    if not os.path.exists(npy_path):
        np.save(npy_path, raw_heat_scores_100)
    if not os.path.exists(raw_npy_path):
        np.save(raw_npy_path, raw_heat_scores_100)
    if not os.path.exists(diff_npy_path):
        np.save(diff_npy_path, diff_heat_scores_100)

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
        # Process BOTH bodies into the XAI scale!
        full_heat_scores_100 = extract_xai_red_dots(attention_matrix[0].cpu().numpy()) # (2, 100, 25)
        
        if actual_frames > 100:
            indices = np.linspace(0, 99, actual_frames).astype(int)
            heat_scores = full_heat_scores_100[:, indices, :]
        else:
            heat_scores = full_heat_scores_100[:, :actual_frames, :]
            
        all_coords = raw_skeleton_tensor[:, :, :, 0:3]

        # 2. DYNAMIC CAMERA BOUNDS (Fixes the blank screen)
        # Ensure we don't include missing joints (marked as 0,0,0) in our camera center calculation
        valid_coords = all_coords[np.any(all_coords != 0, axis=-1)]
        if len(valid_coords) == 0:
            continue # Failsafe if the scene was completely empty
            
        mid_x = np.mean(valid_coords[:, 0])
        mid_y = np.mean(valid_coords[:, 1]) # Kinect Y is Height
        mid_z = np.mean(valid_coords[:, 2]) # Kinect Z is Depth

        fig = plt.figure(figsize=(10, 8)) # Made figure slightly wider to fit colorbar beautifully
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        cmap = plt.get_cmap('coolwarm') 

        # --- COLORBAR ---
        # Representing heatmap values (Blue = Low Attention, Red = High Attention)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.ax.set_title('Attention Level', fontsize=10, pad=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Low (Blue)', 'Medium', 'High (Red)']) 

        def update(frame_idx):
            ax.clear()
            
            # Lock the camera to a perfect 2-meter cube centered exactly on the actor!
            ax.set_xlim(mid_x - 1, mid_x + 1)
            ax.set_ylim(mid_z - 1, mid_z + 1) # Matplotlib Y axis handles Depth
            ax.set_zlim(mid_y - 1, mid_y + 1) # Matplotlib Z axis handles Height
            
            true_action_name = NTU_CLASSES[true_label.item()]
            
            if pred_label == true_label.item():
                status = f"Correct ({pred_confidence:.1f}%)"
            else:
                status = f"Wrong: {NTU_CLASSES[pred_label]} ({pred_confidence:.1f}%)"
                
            ax.set_title(f'Action: {true_action_name}\nAccuracy: {status}')
            ax.set_xlabel('X (Meters)')
            ax.set_ylabel('Depth (Z Meters)')
            ax.set_zlabel('Height (Y Meters)')
            
            # Draw BOTH bodies simultaneously
            for body_idx in range(2):
                current_skeleton = all_coords[frame_idx, body_idx]
                current_heat = heat_scores[body_idx, frame_idx]
                
                # Mask out missing joints (0,0,0)
                valid_joints = np.any(current_skeleton != 0, axis=-1)
                if not np.any(valid_joints):
                    continue # Body missing from this frame
                
                xs, ys, zs = current_skeleton[:, 0], current_skeleton[:, 1], current_skeleton[:, 2]
                colors = cmap(current_heat)
                
                ax.scatter(xs[valid_joints], zs[valid_joints], ys[valid_joints], 
                           c=colors[valid_joints], s=60, edgecolors='black', linewidth=1, zorder=2)
                
                for bone in kinect_bones:
                    j1, j2 = bone
                    if not valid_joints[j1] or not valid_joints[j2]:
                        continue
                    ax.plot([xs[j1], xs[j2]], [zs[j1], zs[j2]], [ys[j1], ys[j2]], c='black', linewidth=2, zorder=1)

        # ==========================================
        # NEW CODE: SAVE THE PEAK ACTION FRAME PNG
        # ==========================================
        # Find the parent results directory based on where the GIF is saving
        os.makedirs(FRAME_DIR, exist_ok=True)
        
        # Build the filename (e.g., A001_peak_frame_47.png)
        base_name = os.path.basename(gif_path).replace('.gif', '')
        png_filename = f"{base_name}_peak_frame_{diff_peak_frame}.png"
        png_path = os.path.join(FRAME_DIR, png_filename)
        
        # Force the plot to draw the exact peak frame, then save it
        update(diff_peak_frame)
        plt.savefig(png_path, bbox_inches='tight', dpi=300, facecolor='white')
        # ==========================================

        # Disable printing inside the loop so it doesn't break the progress bar
        ani = animation.FuncAnimation(fig, update, frames=actual_frames, interval=50)
        ani.save(gif_path, writer='pillow', fps=20)
        
        plt.close(fig)

print("Batch Processing Complete! Check the 'results/xai_npy' and 'results/xai_gifs' folders.")
