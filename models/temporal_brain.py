import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import math

class Temporal_Brain_Layer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_joints=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_node = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        
        # Standard projections for Queries, Keys, and Values
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=0.3)
        
        # 1. Define the 5 Anatomical Rooms (0-indexed)
        # We use a tensor here so the GPU compiler can read it fast
        room_mapping = [
            5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 
            5, 1, 1, 2, 2,  # The 25 joints mapped to rooms 1-5
            0 # DUMMY ROOM FOR PADDING
        ]
        self.register_buffer('room_map', torch.tensor(room_mapping, dtype=torch.int32))
        
        # The Global Node is the 26th token (index 25)
        self.global_node_idx = 25
        self.compiled_self_attention = torch.compile(flex_attention)

        # Pre-build the static block mask once at init.
        # B=None tells flex_attention the mask applies to any batch size (no B=1 broadcast risk).
        # S=26 is fixed (25 joints + 1 global node), so this is safe to cache.
        # Store the mask rule — the actual BlockMask is built lazily on the
        # first forward() call so it is always on the same device as x.
        self._num_heads = num_heads
        self._cached_block_mask = None   # built on first forward pass

        global_node_idx_ref = self.global_node_idx
        def _static_mask_rule(b, h, q_idx, kv_idx):
            is_global_q  = (q_idx == global_node_idx_ref)
            is_global_kv = (kv_idx == global_node_idx_ref)
            same_room = (q_idx < 25) & (kv_idx < 25) & (self.room_map[q_idx] == self.room_map[kv_idx])
            return is_global_q | is_global_kv | same_room
        self._mask_rule = _static_mask_rule


    def forward(self, x, return_attention=False):
        # x shape: (Batch, Time, Sequence_Length, Embed_Dim)
        # Sequence Length here is our 25 joints + 1 Global Node = 26 tokens
        original_shape = x.shape
        if len(original_shape) == 4:
            B, T, S, E = original_shape
            x = x.view(B*T, S, E)
        else:
            B, S, E = original_shape
            T = 1

        # The FlexAttention block mask is built lazily on the first forward pass (see lines 71-75).
        
        # 1. Generate Queries, Keys, and Values
        qkv = self.qkv_proj(x)
        
        # 2. Reshape for Multi-Head Attention: (Batch, Heads, Sequence, Head_Dim)
        qkv = qkv.view(x.shape[0], S, 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 3. Build the block mask on first forward pass (lazy init).
        # This guarantees the mask is on the same device as x (room_map is a
        # registered buffer, so it was already moved by .to(device) before here).
        if self._cached_block_mask is None:
            self._cached_block_mask = create_block_mask(
                self._mask_rule, B=None, H=self._num_heads, Q_LEN=26, KV_LEN=26,
                device=x.device
            )
        block_mask = self._cached_block_mask

        # 4. Execute FlexAttention
        # The GPU will physically skip computing the O(N^2) empty space
        attn_output = self.compiled_self_attention(q, k, v, block_mask=block_mask)
        
        # 5. Reshape and project back out
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(x.shape[0], S, self.embed_dim)
        out = self.out_proj(attn_output)

        out = self.dropout(out)
        
        # === Glass Box ===
        if return_attention:
            # We manually recreate the Q * K^T math just to get the percentages
            d_k = q.size(-1)
            raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Convert the cached Triton hardware mask into a normal PyTorch boolean mask
            dense_mask = self._cached_block_mask.to_dense()
            
            # Mask out the forbidden rooms with -infinity so Softmax turns them into 0%
            raw_scores = raw_scores.masked_fill(~dense_mask.bool(), float('-inf'))
            attn_weights = torch.softmax(raw_scores, dim=-1)
            
            # Average the attention across all 4 heads
            attn_weights_mean = attn_weights.mean(dim=1)
            
            # Reshape back to 4D if necessary
            if len(original_shape) == 4:
                out = out.view(B, T, S, self.embed_dim)
                attn_weights_mean = attn_weights_mean.view(B, T, S, S)
                
            return out, attn_weights_mean

        # Reshape back to 4D for standard training
        if len(original_shape) == 4:
            out = out.view(B, T, S, self.embed_dim)
        
        return out