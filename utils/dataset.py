import os
import torch
import numpy as np
from torch.utils.data import Dataset

class NTUSkeletonDataset(Dataset):
    def __init__(self, data_folder, max_frames=100):
        self.data_folder = os.path.join(data_folder, 'binary_pt') 
        self.file_list = sorted([f for f in os.listdir(self.data_folder) if f.endswith('.pt')])
        self.max_frames = max_frames

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

        return torch.tensor(standardized_tensor, dtype=torch.float32), torch.tensor(action_label, dtype=torch.long)
