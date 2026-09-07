import os
import torch
import numpy as np
from torch.utils.data import Dataset
import math

def random_y_rotation(skeleton_tensor, max_degrees=15):
    """
    Applies a random Y-axis (vertical) rotation to the entire skeleton.
    Input shape: (Time, Bodies, Joints, 3)
    """
    # 1. Pick a random angle between -max_degrees and +max_degrees
    angle = math.radians(np.random.uniform(-max_degrees, max_degrees))
    
    # 2. Build the 3D Rotation Matrix for the Y-axis (Kinect's vertical axis)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot_matrix = np.array([
        [cos_a,  0, sin_a],
        [0,      1, 0    ],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    
    # 3. Apply the rotation to the XYZ coordinates
    # skeleton_tensor is shape (T, M, V, 3). Matrix mutiplication on the last dim.
    rotated_skeleton = np.dot(skeleton_tensor, rot_matrix.T)
    return rotated_skeleton

def engineer_physics_features(skeleton_tensor):
    """
    Transforms raw (X,Y,Z) coordinates into explicit physics features.
    Expected input shape: (Time, Bodies=2, Joints=25, Channels=3)
    Output shape: (Time, Bodies=2, Joints=25, Channels=9)

    """

    # --- THE SOTA FIX: ROOT CENTERING ---
    # Joint 0 is the Base of the Spine. We make it the (0,0,0) origin point.
    # This completely destroys the network's ability to memorize room locations!
    root_joint = skeleton_tensor[:, :, 0:1, :].clone() # Extract spine coords
    skeleton_tensor = skeleton_tensor - root_joint     # Subtract from all 25 joints
    # ------------------------------------

    T, M, V, C = skeleton_tensor.shape

    # 1. Kinematic Tree
    parents = [0, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 21, 24, 23]

    # 2. Calculate Bones
    bones = torch.zeros_like(skeleton_tensor)
    for v in range(V):
        bones[:, :, v, :] = skeleton_tensor[:, :, v, :] - skeleton_tensor[:, :, parents[v], :]
    
    # 3. Calculate Velocity
    velocity = torch.zeros_like(skeleton_tensor)
    velocity[:-1, :, :, :] = skeleton_tensor[1:, :, :, :] - skeleton_tensor[:-1, :, :, :]

    # Stack: Relative (3) + Bones (3) + Velocity (3) = 9 Channels
    engineered_tensor = torch.cat([skeleton_tensor, bones, velocity], dim=-1)

    # return bones
    return engineered_tensor

class NTUSkeletonDataset(Dataset):
    def __init__(self, data_folder, max_frames=100, is_train=False):
        self.data_folder = os.path.join(data_folder, 'binary_pt') 
        self.file_list = sorted([f for f in os.listdir(self.data_folder) if f.endswith('.pt')])
        self.max_frames = max_frames
        self.is_train = is_train

    def __len__(self):
        return len(self.file_list)

    def parse_single_skeleton(self,file_path):
        with open(file_path, 'r') as f:
            datas = f.readlines()
        
        if not datas:
            return None

        nframe = int(datas[0].strip())
        skeleton_tensor = np.zeros((nframe, 2, 25, 3), dtype=np.float32)

        cursor = 0
        for frame in range(nframe):
            cursor += 1
            bodycount = int(datas[cursor].strip())

            if bodycount == 0:
                continue

            for body in range(bodycount):
                cursor += 2 #skip kinect metadata
                njoints_in_file = int(datas[cursor].strip())

                for joint in range(njoints_in_file):
                    cursor += 1
                    if body < 2:
                        joininfo = datas[cursor].strip().split()
                        skeleton_tensor[frame, body, joint, :] = [float(joininfo[0]), float(joininfo[1]), float(joininfo[2])] #appends the (x,y,z)
        
        return skeleton_tensor
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)

        action_string = file_name.split('A')[1][:3]
        action_label = int(action_string) - 1

        raw_tensor = torch.load(file_path, weights_only=True)
        raw_numpy = raw_tensor.numpy()

        actual_frames = raw_numpy.shape[0]
        standardized_tensor = np.zeros((self.max_frames, 2, 25, 3), dtype=np.float32)

        if actual_frames <= self.max_frames:
            standardized_tensor[:actual_frames, :, :, :] = raw_numpy
        else:
            standardized_tensor = raw_numpy[:self.max_frames, :, :, :]

        if self.is_train:
            standardized_tensor = random_y_rotation(standardized_tensor, max_degrees=17)

        tensor_data = torch.tensor(standardized_tensor, dtype=torch.float32)
        engineered_data = engineer_physics_features(tensor_data)

        # Decoupled Body

        # 1. Permute to (Bodies, Time, Joints, Channels)
        engineered_data = engineered_data.permute(1, 0, 2, 3) # (Bodies=2, Time=100, Joints=25, Channels=9)

        # 2. Ghost Mask for Temporal Transformer
        M  = engineered_data.shape[0]
        body_mask = torch.ones(M, dtype=torch.bool)

        for m in range(M):
            # if the body has any kinetic variance mark it as False (not ghost)
            if torch.sum(torch.abs(engineered_data[m])) > 1e-4:
                body_mask[m] = False

        return engineered_data, body_mask, torch.tensor(action_label, dtype=torch.long)