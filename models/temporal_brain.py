import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import math

class Temporal_Brain_Layer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_joints=25,max_frames=100, max_bodies=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        #==== Spatial Components ====
        self.global_node = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.spatial_qkv_proj = nn.Linear(embed_dim, embed_dim*3)
        self.spatial_out_proj = nn.Linear(embed_dim, embed_dim)
        self.spatial_norm1 = nn.LayerNorm(embed_dim)
        self.spatial_norm2 = nn.LayerNorm(embed_dim)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=0.2)
        )

        #==== Temporal Components ====
        # 1. TIME Embedding (Which Frame?)
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, 1, max_frames,embed_dim))

        # 2. Body Embedding (Which Body does this token belong to?)
        self.body_pos_embed = nn.Parameter(torch.randn(1,max_bodies, 1,embed_dim))

        # 3. Video Token (Final Reviewer)
        self.video_token = nn.Parameter(torch.randn(1,1,embed_dim))
        
        #temporal attention layer
        self.temporal_norm1 = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.5, batch_first=True)
        self.temporal_norm2 = nn.LayerNorm(embed_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(p=0.2)
        )

        self.dropout = nn.Dropout(p=0.2)
        
        # 1. Define the 5 Anatomical Rooms (0-indexed)
        room_mapping = [
            5, 5, 5, 5, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 
            5, 1, 1, 2, 2,  # The 25 joints mapped to rooms 1-5
            0 # DUMMY ROOM FOR PADDING
        ]
        self.register_buffer('room_map', torch.tensor(room_mapping, dtype=torch.int32))
        
        # The Global Node is the 26th token (index 25)
        self.global_node_idx = 25
        self.compiled_self_attention = torch.compile(flex_attention)

        self._num_heads = num_heads
        self._cached_block_mask = None   # built on first forward pass

        global_node_idx_ref = self.global_node_idx
        def _static_mask_rule(b, h, q_idx, kv_idx):
            is_global_q  = (q_idx == global_node_idx_ref)
            is_global_kv = (kv_idx == global_node_idx_ref)
            same_room = (q_idx < 25) & (kv_idx < 25) & (self.room_map[q_idx] == self.room_map[kv_idx])
            return is_global_q | is_global_kv | same_room
        self._mask_rule = _static_mask_rule


    def forward(self, x, B, M, body_mask=None, return_attention=False):
        # x -> shape: (B*M, Time, 26, Embed_Dim)
        B_M, T, S, E = x.shape

        #===== BLOCK 1: Spatial Attention (Independent Frames) ====
        x_spatial = x.view(B_M * T, S, E)

        x_norm = self.spatial_norm1(x_spatial)
        qkv = self.spatial_qkv_proj(x_norm)
        qkv = qkv.view(B_M * T, S, 3, self.num_heads, self.embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0].contiguous()
        k = qkv[1].contiguous()
        v = qkv[2].contiguous()

        if self._cached_block_mask is None:
            self._cached_block_mask = create_block_mask(
                self._mask_rule, B=None, H=self._num_heads, Q_LEN = 26, KV_LEN=26, device=x.device
            )
        
        attn_output = self.compiled_self_attention(q, k, v, block_mask=self._cached_block_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B_M * T, S, E)

        attn_output = self.spatial_out_proj(attn_output)
        x_spatial = x_spatial + self.dropout(attn_output)
        x_spatial = x_spatial + self.spatial_ffn(self.spatial_norm2(x_spatial))

        spatial_globals = x_spatial[:, 25, :] #Extract Global Node: (B_M * T, E)

        #==== BLOCK 2: TEMPORAL ATTENTION (Connecting Time & Bodies) ====
        # 1. Unfold to seperate bodies -> Shape: (B, M, T, E)
        x_temporal = spatial_globals.view(B, M, T, E)

        # 2. Inject Identities (Body ID & Time Frame)
        x_temporal = x_temporal + self.body_pos_embed[:, :M, :, :]
        x_temporal = x_temporal + self.temporal_pos_embed[:, :, :T, :]

        # 3. Flatten to (B*M, T, E) for temporal attention
        x_temporal = x_temporal.view(B, M*T, E)

        # 4. Attach Video token to the fornt of the seq. -> Shape: (B_M, T+1, 64)
        video_token_expanded = self.video_token.expand(B, -1, -1)
        x_temporal = torch.cat([video_token_expanded, x_temporal], dim=1)

        # 5. Handle the ghost mask 
        if body_mask is not None:
            # body_mask is (B, M) -> expand to (B, M, T) to cover time
            expanded_mask = body_mask.unsqueeze(-1).expand(B, M, T).reshape(B, M*T)
            # Temporal token is always real, so mask is false for it
            ceo_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            # final mask shape: (B, 1+M*T)
            key_padding_mask = torch.cat([ceo_mask, expanded_mask], dim=1)

        else:
            key_padding_mask = None

        # 6. Temporal attention
        x_temp_norm = self.temporal_norm1(x_temporal)
        temp_attn_out, temporal_attn_weights = self.temporal_attn(
            query=x_temp_norm, key=x_temp_norm, value=x_temp_norm, 
            key_padding_mask=key_padding_mask,
            need_weights=return_attention
        )

        x_temporal = x_temporal + self.dropout(temp_attn_out)
        x_temporal = x_temporal + self.temporal_ffn(self.temporal_norm2(x_temporal))

        # 7. Extract full processed video toke (index 0)
        final_video_representation = x_temporal[:, 0, :]
        
        
        #==== GLASS BOX XAI ====
        if return_attention:
            # 1. Temporal Attention (Which frames matter?)
            video_token_attention = temporal_attn_weights[:, 0, 1:] # shape: (B,M*T)
            video_token_attention = video_token_attention.view(B, M, T) # unflatten to (B, M, T)
            
            # 2. Spatial Attention (Which joints matter?)
            global_q = q[:, :, 25:26, :]
            physical_k = k[:, :, :25, :] # Slice the Keys (k) to ONLY include the 25 physical joints before the math!
            spatial_scores = torch.matmul(global_q, physical_k.transpose(-2, -1)) / math.sqrt(q.size(-1)) # Compute raw scores only against the 25 joints
            spatial_probs = torch.softmax(spatial_scores, dim=-1)
            
            spatial_attention = spatial_probs.max(dim=1).values.squeeze(1) 
            spatial_attention = spatial_attention.view(B, M, T, 25) # Unflatten to (B, M, T, Joints)
            
            # 3. Spatial and Temporal Fused
            # Multiply Joint importance by Frame importance
            ultimate_xai_matrix = spatial_attention * video_token_attention.unsqueeze(-1)
            
            return final_video_representation, ultimate_xai_matrix

        return final_video_representation