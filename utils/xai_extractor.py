import numpy as np

def extract_xai_red_dots(attention_matrix):
    # attention_matrix shape from evaluate.py: (Time, 26, 26)

    # 1. Isolate the Global Node (Index 25) looking at the 25 physical joints (0-24)
    # We slice across ALL frames using ':'
    global_node_weights = attention_matrix[:, 25, :25] # New Shape: (Time, 25)

    # 2. The Math (Min-Max Normalization per frame)
    # keepdims=True ensures the math broadcasts correctly across the Time dimension
    min_val = np.min(global_node_weights, axis=1, keepdims=True)
    max_val = np.max(global_node_weights, axis=1, keepdims=True)

    # We add 1e-8 to prevent dividing by zero
    heat_scores = (global_node_weights - min_val) / (max_val - min_val + 1e-8)
    
    return heat_scores # Final Shape: (Time, 25)

