import torch
import torch.nn as nn

class Spatial_GCN_Layer(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_joints=25):
        super().__init__()

        #Learnable bones -> Shape:(25,25)
        self.adj_matrix = nn.Parameter(torch.randn(num_joints,num_joints))
        self.linear = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self,x):
        # x -> has an input shape of (Batch, Time, Joints, Channels)
        
        #Step A: Expand the features -> e.g 3 channels to 64 channels
        x = self.linear(x)

        #Step B: The Graph Convolution (Message Passing)
        out = torch.einsum('btjc, jk -> btkc', x, self.adj_matrix)

        #Step C: Apply Activation Function
        out = self.relu(out)

        out = self.dropout(out)
        
        return out
