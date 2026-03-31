import os
import torch
import numpy as np
from tqdm import tqdm

def parse_single_skeleton(file_path):
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

def convert_folder(folder_path):
     print(f"Converting {folder_path} to binary PyTorch files...")
     for filename in tqdm(os.listdir(folder_path)):
          if filename.endswith(".skeleton"):
               full_path = os.path.join(folder_path, filename)
               tensor_data = parse_single_skeleton(full_path)

               if tensor_data is not None:
                    save_path = full_path.replace(".skeleton",".pt")
                    torch.save(torch.tensor(tensor_data), save_path)

convert_folder("data/train_skeletons")
convert_folder("data/val_skeletons")
print("Preprocessing has been completed.")