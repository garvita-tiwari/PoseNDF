"""Data analysis datanalysis using pytorch data loader"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb
from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_rotation_6d
from data.data_splits import amass_splits
import glob
class PoseData(Dataset):


    def __init__(self, mode, data_path,batch_size=4, num_workers=3, num_pts=5000):

        self.mode= mode
        self.path = data_path
        self.num_pts = num_pts
        self.data_samples = amass_splits[self.mode]
        self.data_files = glob.glob(self.path+ '/*/*.npz')
        self.data_files = [ds for ds in self.data_files if ds.split('/')[-2] in self.data_samples]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):

        """read pose and distance data"""
        pose_data = np.load(self.data_files[idx])

        # sample poses from this file 
        subsample_indices = np.random.randint(0, len(pose_data['pose']), self.num_pts)
        poses = pose_data['pose'][subsample_indices]
        dist = np.mean(pose_data['dist'][subsample_indices],axis=1)
        nn_pose = pose_data['nn_pose'][subsample_indices]

        betas = np.zeros(10)


        return {'pose': np.array(poses, dtype=np.float32),
                'dist': np.array(dist, dtype=np.float32),
                'nn_pose': np.array(nn_pose, dtype=np.float32),
                'betas': np.array(betas, dtype=np.float32)}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)



class PoseData_online(Dataset):
    """not used"""


    def __init__(self, mode, data_path,batch_size=1024, num_workers=3, sigma=0.01):

        self.mode= mode
        self.path = data_path
        self.sigma = sigma
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_datasets()

    def load_datasets(self):
        pose_arr = []
        for dset_name in amass_splits[self.mode][:1]:
            npz_fnames = sorted(os.listdir(os.path.join(self.path, dset_name)))
            for npz_fname in npz_fnames[:1]:
                pose_data = np.load(os.path.join(self.path,dset_name, npz_fname))['pose_body']
                pose_arr.extend(axis_angle_to_quaternion(torch.from_numpy(pose_data.reshape(len(pose_data), 21,3)).unsqueeze(0)))
        self.pose = torch.cat(pose_arr)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):


        poses = self.pose[idx] + self.sigma*np.random.rand(21,4)

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

