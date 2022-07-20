"""Data analysis datanalysis using pytorch data loader"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb
from data_splits import amass_splits
from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_rotation_6d
class PoseData(Dataset):


    def __init__(self, data_path, amass_splits, batch_size=1024, num_workers=3):


        self.path = data_path
        self.datasets= amass_splits

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_datasets()

    def load_datasets(self):
        pose_arr = []
        for dset_name in self.datasets:
            npz_fnames = sorted(os.listdir(os.path.join(self.path, dset_name)))
            for npz_fname in npz_fnames:
                pose_arr.extend(np.load(os.path.join(self.path,dset_name, npz_fname))['pose_body'])
        self.pose = np.array(pose_arr)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, idx):


        poses = self.pose[idx]

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




if __name__ == "__main__":

    amass_path = '/BS/humanpose/static00/data/PoseNDF_raw/smpl_h'
    modes = ['train', 'vald', 'test']
    mean = {}
    var = {}
    rep = 'rot6d'
    for mode in modes:
        dataloader_amass = PoseData(amass_path, amass_splits[mode], batch_size=1024)
        pose_data = torch.from_numpy(dataloader_amass.pose)
        if rep == 'aa':
            mean_pose = torch.mean(pose_data, dim=0)
            var_pose = torch.var(pose_data, dim=0)
        if rep == 'quat':
            pose_data = axis_angle_to_quaternion(pose_data.reshape(len(pose_data), 21, 3)).reshape(len(pose_data), 21, 4)

            mean_pose = torch.mean(pose_data, dim=0)
            var_pose = torch.var(pose_data, dim=0)
        if rep == 'mat':
            pose_data = axis_angle_to_matrix(pose_data.reshape(len(pose_data), 21, 3)).reshape(len(pose_data), 21, 9)

            mean_pose = torch.mean(pose_data, dim=0)
            var_pose = torch.var(pose_data, dim=0)
        if rep == 'rot6d':
            pose_data = matrix_to_rotation_6d(axis_angle_to_matrix(pose_data.reshape(len(pose_data), 21, 3))).reshape(len(pose_data), 21, 6)

            mean_pose = torch.mean(pose_data, dim=0)
            var_pose = torch.var(pose_data, dim=0)
        mean[mode +'_mean'] = mean_pose.detach().numpy()
        mean[mode +'_var'] = var_pose.detach().numpy()
    np.savez(os.path.join(amass_path, 'mean_{}.npz'.format(rep)),**mean)