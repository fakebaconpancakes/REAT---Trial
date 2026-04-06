import numpy as np

def extract_xai_red_dots(attention_matrix):
    # attention_matrix shape from our new Spatio-Temporal fusion: (Time, 25)

    # The Math (Global Min-Max Normalization across the WHOLE video)
    # By removing axis=1, we ensure that unimportant frames stay dark, 
    # and only the true peak action of the video hits 1.0 (Red Hot).
    min_val = np.min(attention_matrix)
    max_val = np.max(attention_matrix)

    # We add 1e-8 to prevent dividing by zero
    heat_scores = (attention_matrix - min_val) / (max_val - min_val + 1e-8)
    
    return heat_scores # Final Shape: (Time, 25)