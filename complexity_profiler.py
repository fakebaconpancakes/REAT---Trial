import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer

print("Initializing Hardware Complexity Profiler..")

# 1. Model intialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcn = Spatial_GCN_Layer().to(device)
transformer = Temporal_Brain_Layer().to(device)
global_node = nn.Parameter(torch.zeros(1, 1, 1, 64, device=device))
classifier = nn.Linear(64,60).to(device)

# 2. Dummy Video
B, M, T, V, C = 1, 2, 100, 25, 3
dummy_input = torch.randn(B, M, T, V, C).to(device) 
gcn_input = dummy_input.permute(0, 1, 2, 3, 4).reshape(-1, T, V, C) # Shape: (2,100,25,3)

# 3. Profile the GCN
print("(1) Profiling GCN...")
gcn_macs, gcn_params = profile(gcn, inputs=(gcn_input, ), verbose=False)

# 4. Profile Temporal Brain
with torch.no_grad():
    gcn_features = gcn(gcn_input)

frames = gcn_features.shape[1]
global_node_expanded = global_node.expand(B*M, frames, 1, 64)
transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)

print("(2) Profile Temporal Brain...")
tf_macs, tf_params = profile(transformer, inputs=(transformer_input, False), verbose=False)

# ==========================================
# 🚨 MANUAL FLEX_ATTENTION MACS CORRECTION 🚨
# ==========================================
# thop ignores the functional flex_attention call, so we calculate its MACs manually.
# Your notebook confirmed your mask has 180 active connections.
active_connections = 180 
batch_size = B * M  # 2 bodies
time_steps = T      # 100 frames
embed_dim = 64

# Math: (Q * K^T MACs) + (Attn * V MACs) = 2 * active_connections * embed_dim
flex_attn_macs = batch_size * time_steps * 2 * active_connections * embed_dim

# Add the missing math back into the Transformer's total
tf_macs += flex_attn_macs
# ==========================================

# 5. Profile the Classifier
with torch.no_grad():
    attn_output = transformer(transformer_input)
    final_features = attn_output[:, -1, 25, :]
    separated = final_features.view(B, M, 64)
    video_rep, _ = torch.max(separated, dim=1)

print("(3) Profile Classifier...")
class_macs, class_params = profile(classifier, inputs=(video_rep, ), verbose=False)

# 6. Calculate Totals and Format
total_macs = gcn_macs + tf_macs + class_macs
total_params = gcn_params + tf_params + class_params

# MACs (Multiply-Accumulate Operations) are usually multiplied by 2 to get standard FLOPs
total_flops = total_macs * 2

formatted_flops, formatted_params = clever_format([total_flops, total_params], "%.2f")

print("-" * 50)
print("ARCHITECTURE COMPLEXITY REPORT (Single-Stream)")
print("-" * 50)
print(f"Total Parameters: {formatted_params} (VRAM Footprint)")
print(f"Total FLOPs:      {formatted_flops} (Compute Cost)")
print("-" * 50)

