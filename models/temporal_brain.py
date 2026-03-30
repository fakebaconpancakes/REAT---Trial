import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

class Temporal_Brain_Layer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_joints=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Standard projections for Queries, Keys, and Values
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 1. Define the 5 Anatomical Rooms (0-indexed)
        # We use a tensor here so the GPU compiler can read it fast
        room_mapping = [
            5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 
            5, 1, 1, 2, 2  # The 25 joints mapped to rooms 1-5
        ]
        self.register_buffer('room_map', torch.tensor(room_mapping, dtype=torch.int32))
        
        # The Global Node is the 26th token (index 25)
        self.global_node_idx = 25 

    # 2. The FlexAttention Masking Rule
    # This function tells the hardware exactly which tokens are allowed to communicate
    def anatomical_mask_rule(self, b, h, q_idx, kv_idx):
        # b = batch, h = head, q_idx = "Who is looking", kv_idx = "Who is being looked at"
        
        # Rule A: The Project Manager. If either token is the Global Node, ALLOW connection.
        is_global_q = (q_idx == self.global_node_idx)
        is_global_kv = (kv_idx == self.global_node_idx)
        
        # Rule B: The Soundproof Rooms. Do they share the same room ID?
        # We add a safety check (< 25) to avoid indexing out of bounds for the global node
        same_room = (q_idx < 25) & (kv_idx < 25) & (self.room_map[q_idx] == self.room_map[kv_idx])
        
        # Final Verdict: Allow connection if it's the global node, OR if they are in the same room.
        return is_global_q | is_global_kv | same_room

    def forward(self, x):
        # x shape: (Batch, Sequence_Length, Embed_Dim)
        # Sequence Length here is our 25 joints + 1 Global Node = 26 tokens
        B, S, E = x.shape 
        
        # 1. Generate Queries, Keys, and Values
        qkv = self.qkv_proj(x)
        
        # 2. Reshape for Multi-Head Attention: (Batch, Heads, Sequence, Head_Dim)
        qkv = qkv.view(B, S, 3, self.num_heads, E // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. Compile the Hardware-Sparse Mask!
        # This is where the Triton compiler writes the custom instruction for the RTX 3090
        block_mask = create_block_mask(self.anatomical_mask_rule, B, self.num_heads, S, S)
        
        # 4. Execute FlexAttention
        # The GPU will physically skip computing the O(N^2) empty space
        attn_output = flex_attention(q, k, v, block_mask=block_mask)
        
        # 5. Reshape and project back out
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, S, E)
        out = self.out_proj(attn_output)
        
        return out