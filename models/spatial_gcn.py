import torch
import torch.nn as nn
import numpy as np

def get_normalized_biological_matrix(num_joints=25):
    # 1. Define the true human bone connections for NTU RGB+D (0-indexed)
    # The connections are based on the standard 25-joint Kinect v2 skeleton
    neighbor_link = [
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
        (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
        (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
    ]
    
    # 2. Build the basic Adjacency Matrix (A)
    A = np.zeros((num_joints, num_joints))
    for i, j in neighbor_link:
        # Subtract 1 to make it 0-indexed for Python arrays
        A[i - 1, j - 1] = 1
        A[j - 1, i - 1] = 1
        
    # 3. Add the Self-Loops (A + I)
    A = A + np.eye(num_joints)
    
    # 4. Calculate the Degree Matrix (D) and Normalize
    # Formula: D^(-1/2) * A * D^(-1/2)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0 # Prevent division by zero errors
    
    # Create diagonal matrix for D^(-1/2)
    D_mat = np.diag(D_inv_sqrt)
    
    # Multiply the matrices
    normalized_A = np.dot(np.dot(D_mat, A), D_mat)
    
    return torch.tensor(normalized_A, dtype=torch.float32)

class Spatial_GCN_Layer(nn.Module):
    def __init__(self, in_channels=9, out_channels=64, num_joints=25):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        biological_matrix = get_normalized_biological_matrix(num_joints)
        self.register_buffer('adj_matrix', biological_matrix)
    
    def forward(self,x):
        # x -> has an input shape of (Batch, Time, Joints, Channels)
        
        #Step A: Expand the features -> e.g 3 channels to 64 channels
        x_proj = self.linear(x)

        #Step B: The Graph Convolution (Message Passing)
        out = torch.einsum('btjc, jk -> btkc', x_proj, self.adj_matrix)

        #Step C: The Residual Connection + Norm
        out = self.norm(out + x_proj)

        #Step D: Apply Activation Function
        out = self.relu(out)
        out = self.dropout(out)
        
        return out
