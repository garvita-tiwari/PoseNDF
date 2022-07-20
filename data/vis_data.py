"""Visualize mean data"""
import smplx
import torch
import numpy as np
import os
import sys
sys.path.append('/BS/garvita/work/code/posendf/visualisation')
from blender_renderer_pytorch import visualize_body
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.transforms import quaternion_to_axis_angle, matrix_to_quaternion, rotation_6d_to_matrix


if __name__ == "__main__":

    amass_path = '/BS/humanpose/static00/data/PoseNDF_raw/smpl_h'
    modes = ['train', 'vald', 'test']
    mean = {}
    var = {}
    rep = 'aa'
    modes = ['train', 'vald', 'test']
    device = 'cpu'

    for mode in modes:

        pose_body = torch.from_numpy(np.load(os.path.join(amass_path, 'mean_{}.npz'.format(rep)))['{}_mean'.format(mode)]).unsqueeze(0)
        if rep == 'quat':
            pose_body = quaternion_to_axis_angle(pose_body.reshape(len(pose_body), 21, 4)).reshape(len(pose_body),63)


        if rep == 'mat':
            pose_body = quaternion_to_axis_angle(matrix_to_quaternion(pose_body.reshape(len(pose_body), 21, 3,3))).reshape(len(pose_body), 63)

        if rep == 'rot6d':
            pose_body = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(pose_body.reshape(len(pose_body), 21, 6)))).reshape(len(pose_body), 63)


        hand_pose = torch.zeros((len(pose_body), 6)).to(device)
        body_pose = torch.cat((pose_body, hand_pose), dim=1)
        model_folder = '/BS/garvita/work/SMPL_data/models'  #
        batch_size = 1
        body_model = smplx.create(model_folder, model_type='smpl', gender='male', num_betas=10,
                                       batch_size=batch_size)  # we are running all the datasets for male gendern
        body_model = body_model.to(device=device)
        beta = np.zeros((10))
        betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0).repeat(batch_size, 1).to(device)
        smpl_posed = body_model.forward(betas=betas_torch, body_pose=body_pose)

        verts = smpl_posed['vertices']
        if os.path.exists(os.path.join(amass_path, "{}_{}_mean.png".format(mode, rep))):
            print('already done................................................')
        verts = smpl_posed['vertices'][0] + torch.from_numpy(np.array([0, 0.5, 0]))
        min_z = torch.min(verts[:, 1])
        # body  = Meshes(verts=[verts], faces=[body_model.faces_tensor])
        body_path = amass_path + "/body.obj"

        save_obj(body_path, verts, body_model.faces_tensor)

        visualize_body(body_path, min_z, os.path.join(amass_path,"{}_{}_mean.png".format(mode, rep)),
                       side='front', out_folder=amass_path)