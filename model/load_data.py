"""Data analysis datanalysis using pytorch data loader"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb
from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_rotation_6d

class PoseData(Dataset):


    def __init__(self, mode, data_path, amass_splits, batch_size=1024, num_workers=3):


        self.path = data_path
        self.datasets= amass_splits

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_datasets()

    def load_datasets(self):
        pose_arr = []
        for dset_name in self.datasets[:1]:
            npz_fnames = sorted(os.listdir(os.path.join(self.path, dset_name)))
            for npz_fname in npz_fnames[:1]:
                pose_data = np.load(os.path.join(self.path,dset_name, npz_fname))['pose_body']
                pose_arr.extend(axis_angle_to_quaternion(torch.from_numpy(pose_data.reshape(len(pose_data), 21,3)).unsqueeze(0)))
        self.pose = torch.cat(pose_arr)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):


        poses = self.pose[idx] + np.random.rand(21,4)

        betas = np.zeros(10)


        return {'pose': np.array(poses, dtype=np.float32),
                'pose_id': idx,
                'betas': np.array(betas, dtype=np.float32)}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

