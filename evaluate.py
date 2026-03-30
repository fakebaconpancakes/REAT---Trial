import os
import torch
import torch.nn as nn
import numpy as np
from  torch.utils.data import DataLoader
from models import spatial_gcn, temporal_brain
from utils.dataset import NTUSkeletonDataset

# 1. HARDWARE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in-use: {device.type.upper()}")