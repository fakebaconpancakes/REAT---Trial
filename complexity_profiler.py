import torch
import torch.nn as nn
from thop import profile, clever_format

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer

# 1. Wrap the entire pipeline into one continuous model for profiling
class REAT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = Spatial_GCN_Layer(in_channels=9, out_channels=64)
        self.transformer = Temporal_Brain_Layer(embed_dim=64, num_heads=4, max_frames=100)
        
    def forward(self, x):
        # x shape: (Batch, Time, Bodies, Joints, Channels) -> (1, 100, 2, 25, 9)
        B, T, M, V, C = x.shape
        gcn_input = x.permute(0, 2, 1, 3, 4).reshape(-1, T, V, C) # Shape: (2, 100, 25, 9)
        
        gcn_features = self.gcn(gcn_input) # Shape: (2, 100, 25, 64)
        
        # Extract and concatenate the Global Node
        global_node = self.transformer.global_node.expand(B*M, T, 1, 64)
        transformer_input = torch.cat([gcn_features, global_node], dim=2)
        
        # Forward pass through the Temporal Brain (with the FFN!)
        out = self.transformer(transformer_input)
        return out

# 2. Initialize and Profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Building REAT Profiler on {device}...")
model = REAT_Model().to(device)

# Create a dummy tensor that perfectly matches your 9-channel Physics Engine output
# (Batch=1, Time=100, Bodies=2, Joints=25, Channels=9)
dummy_input = torch.randn(1, 100, 2, 25, 9).to(device)

print("Calculating MACs (Compute) and Parameters (Memory)...")
# verbose=False stops it from printing every single internal PyTorch operation
macs, params = profile(model, inputs=(dummy_input, ), verbose=False)

# Convert huge raw numbers to human-readable format (e.g., "M" for Mega, "G" for Giga)
macs_str, params_str = clever_format([macs, params], "%.3f")

print("\n" + "="*50)
print(f"REAT Architecture Complexity Report")
print("="*50)
print(f"Total Parameters:  {params_str}")
print(f"Total MACs:        {macs_str}")
print("="*50)
print("*Note: 1 MAC (Multiply-Accumulate) ≈ 2 FLOPs")