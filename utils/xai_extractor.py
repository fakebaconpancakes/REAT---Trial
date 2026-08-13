import numpy as np 

def extract_xai_red_dots(attention_matrix): 
    # attention_matrix shape from our Decoupled Spatio-Temporal fusion: (Bodies, Time, 25) 
    
    # The Math: Global Min-Max Normalization across the WHOLE video and ALL bodies.
    # By omitting the 'axis' parameter, we ensure that:
    # 1. Unimportant frames stay dark.
    # 2. Inactive background bodies stay completely dark.
    # 3. Only the true peak action of the primary actor hits 1.0 (Red Hot).
    
    min_val = np.min(attention_matrix) 
    max_val = np.max(attention_matrix) 
    
    # We add 1e-8 to prevent dividing by zero 
    heat_scores = (attention_matrix - min_val) / (max_val - min_val + 1e-8) 
    
    return heat_scores # Final Shape: (Bodies, Time, 25)