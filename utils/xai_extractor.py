import numpy as np

def extract_xai_red_dots(attention_matrix):
    # attention_matrix.shape -> (26,26)

    # 1. Isolate Global Node -> So only see the connections of 0-24 (The physical joints)
    global_node_weights = attention_matrix[25, :25]

    # 2. The Math (Min-Max Normalization)
    # The raw attention numbers can be very small decimals. 
    # We normalize them so the least important joint is exactly 0.0 (Cold) 
    # and the most important joint is exactly 1.0 (Red Hot).
    min_val = np.min(global_node_weights)
    max_val = np.max(global_node_weights)

    # We add 1e-8 (a tiny number) to prevent dividing by zero if the array is flat
    heat_scores = (global_node_weights - min_val) / (max_val - min_val + 1e-8)
    
    return heat_scores
