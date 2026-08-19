import torch
import torch.nn as nn

try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer


class REAT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = Spatial_GCN_Layer(in_channels=9, out_channels=128)
        # Updated to include max_bodies for the new Body Positional Embeddings
        self.transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2)

    def forward(self, x):
        # x shape: (B, M, T, V, C) -> The new Decoupled Shape
        B, M, T, V, C = x.shape
        
        # The Folding Trick
        gcn_input = x.reshape(B * M, T, V, C)
        gcn_features = self.gcn(gcn_input)
        
        frames = gcn_features.shape[1]
        global_node_expanded = self.transformer.global_node.expand(B * M, frames, 1, 128)
        transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)
        
        # The Interaction Call (passing B and M natively)
        return self.transformer(transformer_input, B, M)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _linear_macs(num_tokens, in_dim, out_dim):
    return num_tokens * in_dim * out_dim


def _ffn_macs(num_tokens, embed_dim, expansion=4):
    hidden = embed_dim * expansion
    return _linear_macs(num_tokens, embed_dim, hidden) + _linear_macs(num_tokens, hidden, embed_dim)


def _spatial_mask_pairs(room_map, global_idx):
    room_map = room_map.tolist()
    total_pairs = 0
    seq_len = len(room_map)
    for q_idx in range(seq_len):
        for kv_idx in range(seq_len):
            is_global_q = q_idx == global_idx
            is_global_kv = kv_idx == global_idx
            same_room = q_idx < 25 and kv_idx < 25 and room_map[q_idx] == room_map[kv_idx]
            if is_global_q or is_global_kv or same_room:
                total_pairs += 1
    return total_pairs


def estimate_reat_macs(model, x_shape):
    # x_shape: (B, M, T, V, C)
    B, M, T, V, C = x_shape
    E = model.transformer.embed_dim
    H = model.transformer.num_heads
    S = V + 1  # 25 joints + 1 global node
    d = E // H
    B_M = B * M

    # ---- Spatial GCN ----
    tokens_gcn = B_M * T * V
    macs_gcn_linear = _linear_macs(tokens_gcn, C, E)
    macs_gcn_einsum = B_M * T * V * E * V

    # ---- Temporal Brain / Spatial block ----
    # Spatial Block processes bodies INDEPENDENTLY (Batch * Bodies)
    tokens_spatial = B_M * T * S
    macs_spatial_qkv = _linear_macs(tokens_spatial, E, 3 * E)

    mask_pairs_per_frame_head = _spatial_mask_pairs(
        model.transformer.room_map, model.transformer.global_node_idx
    )
    macs_spatial_attn = B_M * T * H * mask_pairs_per_frame_head * (2 * d)

    macs_spatial_out = _linear_macs(tokens_spatial, E, E)
    macs_spatial_ffn = _ffn_macs(tokens_spatial, E, expansion=4)

    # ---- Temporal block (UPDATED FOR INTERACTION SEQUENCE) ----
    # Temporal block processes bodies TOGETHER in one flattened sequence.
    # New sequence length = (Bodies * Frames) + 1 Video CEO Token
    L = (M * T) + 1  
    tokens_temporal = B * L 
    
    macs_temporal_qkv = _linear_macs(tokens_temporal, E, 3 * E)
    
    # Attention scales quadratically with sequence length: L * L
    dense_pairs = L * L
    macs_temporal_attn = B * H * dense_pairs * (2 * d)
    
    macs_temporal_out = _linear_macs(tokens_temporal, E, E)
    macs_temporal_ffn = _ffn_macs(tokens_temporal, E, expansion=4)

    total_macs = (
        macs_gcn_linear
        + macs_gcn_einsum
        + macs_spatial_qkv
        + macs_spatial_attn
        + macs_spatial_out
        + macs_spatial_ffn
        + macs_temporal_qkv
        + macs_temporal_attn
        + macs_temporal_out
        + macs_temporal_ffn
    )

    return {
        "total": total_macs,
        "breakdown": {
            "gcn_linear": macs_gcn_linear,
            "gcn_einsum": macs_gcn_einsum,
            "spatial_qkv_proj": macs_spatial_qkv,
            "spatial_attention_sparse": macs_spatial_attn,
            "spatial_out_proj": macs_spatial_out,
            "spatial_ffn": macs_spatial_ffn,
            "temporal_qkv_proj": macs_temporal_qkv,
            "temporal_attention_dense": macs_temporal_attn,
            "temporal_out_proj": macs_temporal_out,
            "temporal_ffn": macs_temporal_ffn,
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building REAT Profiler on {device}...")
    model = REAT_Model().to(device)

    # Updated Dummy Shape: (Batch=1, Bodies=2, Time=100, Joints=25, Channels=9)
    input_shape = (1, 2, 100, 25, 9)
    dummy_input = torch.randn(*input_shape).to(device)

    print("Calculating analytic MACs and exact trainable parameters...")
    analytic = estimate_reat_macs(model, input_shape)
    exact_params = count_trainable_params(model)

    print("\n" + "=" * 60)
    print("REAT Architecture Complexity Report (Analytic)")
    print("=" * 60)
    print(f"Trainable Parameters (exact): {exact_params:,}")
    print(f"Total MACs (analytic):        {analytic['total']:,}")
    print(f"Estimated FLOPs (~2*MACs):    {2 * analytic['total']:,}")
    print("=" * 60)
    print("MAC breakdown:")
    for name, value in analytic["breakdown"].items():
        print(f"  - {name:25s}: {value:,}")

    if profile is not None and clever_format is not None:
        print("\nCalculating THOP result for comparison...")
        thop_macs, thop_params = profile(model, inputs=(dummy_input,), verbose=False)
        thop_macs_str, thop_params_str = clever_format([thop_macs, thop_params], "%.3f")
        delta = analytic["total"] - thop_macs
        delta_pct = (delta / analytic["total"]) * 100 if analytic["total"] else 0.0

        print("\n" + "=" * 60)
        print("THOP Cross-check")
        print("=" * 60)
        print(f"THOP Parameters:            {thop_params_str}")
        print(f"THOP MACs:                  {thop_macs_str}")
        print(f"Analytic - THOP (raw MACs): {int(delta):,} ({delta_pct:.2f}%)")
        print("=" * 60)
        print("*THOP may undercount custom kernels (einsum/flex_attention).")
    else:
        print("\nTHOP is not installed; skipped THOP cross-check.")


if __name__ == "__main__":
    main()