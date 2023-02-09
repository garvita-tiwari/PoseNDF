import argparse
from configs.config import load_config
# General config

#from model_quat import  train_manifold2 as train_manifold
from model.posendf import PoseNDF
import shutil
from data.data_splits import amass_splits
import ipdb
import torch
import numpy as np
from body_model import BodyModel
from exp_utils import renderer, PerspectiveCamera

from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_quaternion
from tqdm import tqdm
from pytorch3d.io import save_obj
import os

class ImageFit(object):
    def __init__(self, posendf,  body_model, out_path='./experiment_results/motion_denoise', debug=False, device='cuda:0', batch_size=1, gender='male', use_joints_conf=True):
        self.debug = debug
        self.device = device
        self.pose_prior = posendf
        self.body_model = body_model
        self.out_path = out_path
        self.use_joints_conf = use_joints_conf
        self.dtype=torch.float32
        init_joints_idxs = [9, 12, 2, 5]
        self.init_joints_idxs = torch.tensor(init_joints_idxs, device=self.device)
        self.trans_estimation = 10.0

    
    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'temp': lambda cst, it: 10. ** 2 * cst * (1 + it),
                       'data': lambda cst, it: 10. ** 1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def visualize(vertices, faces, out_path, device, joints=None, render=False, init=False, ):
        # save meshes and rendered results if needed
        os.makedirs(out_path,exist_ok=True)
        os.makedirs( os.path.join(out_path, 'meshes'), exist_ok=True)
        if init:
            [save_obj(os.path.join(out_path, 'meshes', 'init_{:04}.obj'.format(i) ), vertices[i], faces) for i in range(len(vertices))]
        else:
            [save_obj(os.path.join(out_path, 'meshes', 'results_{:04}.obj'.format(i) ),vertices[i], faces) for i in range(len(vertices))]

        if render:
            renderer(vertices, faces, out_path, device=device, init=init)
        
    def camera_loss(self, camera, body_model_output, gt_joints):
        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.loss_weight['data'] ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss

            
    def projection_loss(self, camera, body_model_output, gt_joints):
        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(gt_joints -projected_joints, 2)
        joint_loss = torch.sum(joint_error) * self.loss_weight['data'] ** 2


        return joint_loss 

    def optimize(self, image, keypoints,  iterations=10, steps_per_iter=10):
        #  create camers with given focal length 
        focal_length = 5000.
        camera = PerspectiveCamera(focal_length_x=focal_length, focal_length_y=focal_length)
        camera = camera.to(device=self.device)

        # create initial SMPL from mean pose and shape
        betas = torch.zeros((batch_size,10)).to(device=self.device)
        pose_body = torch.zeros((batch_size,69)).to(device=self.device)
        smpl_init = self.body_model(betas=betas, pose_body=pose_body) 

        keypoint_data = torch.tensor(keypoints, dtype=self.dtype)
        gt_joints = keypoint_data[:, :, :2].to(device=self.device, dtype=self.dtype)
        if self.use_joints_conf:
            joints_conf = keypoint_data[:, :, 2].reshape(1, -1).to(device=self.device, dtype=self.dtype)

        # Step 1: The indices of the joints used for the initialization of the camera
        camera.translation.requires_grad = True
        smpl_init.global_orient  = True
        smpl_init.pose_body  = False
        smpl_init.betas  = False
        camera_opt_params = [camera.translation, smpl_init.global_orient]
        optimizer = torch.optim.Adam(camera_opt_params, 0.02, betas=(0.9, 0.999))


        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing camera translation for torso joints')
            for i in loop:
                optimizer.zero_grad()
                loss_dict = dict()

                smpl_init = self.body_model(betas=self.betas, pose_body=smpl_init.body_pose, global_orient=smpl_init.global_orient)

                # Get total loss for backward pass
                tot_loss = self.camera_loss(camera, smpl_init, gt_joints)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.8f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)


        # Step 2: Optimize for all joints
        smpl_init.global_orient  = True
        smpl_init.pose_body  = True
        smpl_init.betas  = True
        body_opt_params = [smpl_init.pose_body, smpl_init.global_orient, smpl_init.betas ]
        optimizer = torch.optim.Adam(body_opt_params, 0.02, betas=(0.9, 0.999))


        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing full model')
            for i in loop:
                optimizer.zero_grad()
                loss_dict = dict()

                smpl_init = self.body_model(betas=self.betas, pose_body=smpl_init.body_pose, global_orient=smpl_init.global_orient)

                # convert pose to quaternion and  predict distance
                pose_quat = axis_angle_to_quaternion(smpl_init.body_pose.view(-1, 23, 3)[:, :21])
                loss_dict['pose_pr']= torch.mean(self.pose_prior(pose_quat, train=False)['dist_pred'])

                # Get total loss for backward pass
                loss_dict['data'] = self.projection_loss(camera, smpl_init, gt_joints)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.8f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)


        smpl_init = self.body_model(betas=betas, pose_body=pose_body) 
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True, init=True)

        init_joints = torch.from_numpy(smpl_init.Jtr.detach().cpu().numpy().astype(np.float32)).to(device=self.device)
        init_pose = noisy_poses.detach().cpu().numpy()
        # Optimizer
        smpl_init.body_pose.requires_grad= True
        smpl_init.betas.requires_grad = True
        
        optimizer = torch.optim.Adam([smpl_init.body_pose], 0.02, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL poses')
            for i in loop:
                optimizer.zero_grad()
                loss_dict = dict()
                # convert pose to quaternion and  predict distance
                pose_quat = axis_angle_to_quaternion(smpl_init.body_pose.view(-1, 23, 3)[:, :21])
                loss_dict['pose_pr']= torch.mean(self.pose_prior(pose_quat, train=False)['dist_pred'])

                # calculate temporal loss between mesh vertices
                smpl_init = self.body_model(betas=self.betas, pose_body=smpl_init.body_pose)
                # smpl_opt = self.smpl(betas=self.betas, pose_body=smpl_init.body_pose)
                temp_term = smpl_init.vertices[:-1] - smpl_init.vertices[1:]
                loss_dict['temp'] = torch.mean(torch.sqrt(torch.sum(temp_term*temp_term, dim=2)))

                # calculate data term from inital noisy pose
                if it > 0: #for nans
                    data_term = smpl_init.Jtr  - init_joints
                    loss_dict['data'] = torch.mean(torch.sqrt(torch.sum(data_term*data_term, dim=2)))   

                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()


                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.8f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)


        # ipdb.set_trace()
        # create final results
        smpl_init = self.body_model(betas=self.betas, pose_body=smpl_init.body_pose)
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr, render=True)

        print('** Optimised pose **')

def main(opt, ckpt, image_folder, out_path):
    batch_size = 120
    ### load the model
    net = PoseNDF(opt)
    device= 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)

    #  load body model
    bm_dir_path = '/BS/garvita/work/SMPL_data/models/smpl'
    body_model = BodyModel(bm_path=bm_dir_path, model_type='smpl', batch_size=batch_size,  num_betas=10).to(device=device)

    # load image and keypoints
    motion_data = np.load(motion_file)['pose_body'][:batch_size]
    noisy_poses = torch.from_numpy(motion_data.astype(np.float32)).to(device=device)

    # create Motion denoiser layer
    motion_denoiser = ImageFit(net, body_model=body_model, batch_size=1, out_path=out_path)
    motion_denoiser.optimize(image, keypoints)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interpolate using PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='configs/amass.yaml', type=str, help='Path to config file.')
    parser.add_argument('--ckpt_path', '-ckpt', default='/BS/humanpose/static00/pose_manifold/amass/test_lrelu_l1_1e-05_dist10.0_eik1.0/checkpoints/checkpoint_epoch_best.tar', type=str, help='Path to pretrained model.')
    parser.add_argument('--image_folder', '-if', default='/BS/humanpose/static00/data/PoseNDF_exp/motion_denoise_data/SSM_synced/20161014_50033/punch_kick_sync_poses.npz', type=str, help='Path to image and its keypoint')
    parser.add_argument('--outpath_folder', '-out', default='/BS/humanpose/static00/data/PoseNDF_exp/motion_denoise_results/SSM_synced/20161014_50033/punch_kick_sync_poses', type=str, help='Path to output')
    args = parser.parse_args()

    opt = load_config(args.config)

    main(opt, args.ckpt_path, args.image_folder, args.outpath_folder)