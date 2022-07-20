"""Data analysis datanalysis using pytorch data loader"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb
from data_splits import amass_splits
from pytorch3d.transforms import axis_angle_to_quaternion, axis_angle_to_matrix, matrix_to_rotation_6d
import smplx
if __name__ == "__main__":

    amass_path = '/BS/humanpose/static00/data/PoseNDF_raw/smpl_h_anaylsis'
    mode  = 'train'
    rep = 'quat'
    data_dis = np.load(os.path.join(amass_path, 'mean_{}.npz'.format(rep)))
    mean = torch.from_numpy(data_dis['train_mean'].astype(np.float32))
    sd = torch.from_numpy(np.sqrt(data_dis['train_var']).astype(np.float32))
    var = torch.from_numpy(data_dis['train_var'].astype(np.float32))

    batch_size = 120
    data_name = 'amass_noise_0.5'
    mean= mean.unsqueeze(0).repeat(batch_size,1,1)
    var= var.unsqueeze(0).repeat(batch_size,1,1)
    # data_name = 'amass_clean'
    # data_name = 'partial_1'
    device = 'cpu'
    # model_folder = '/BS/garvita/work/SMPL_data/models'  #
    # body_model = smplx.build_layer(model_folder, model_type='smpl', gender='male', num_betas=10,
    #                                batch_size=batch_size)  # we are running all the datasets for male gendern
    # body_model = body_model.to(device=device)
    # beta = np.zeros((10))
    # betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0).repeat(batch_size, 1).to(device)
    # hand_pose = torch.zeros((batch_size, 2, 3)).to(device)
    all_err_obs = []
    all_err_humor = []
    all_err_vposer = []
    all_sd_var =[]
    all_err_im = []
    model_folder = '/BS/garvita/work/SMPL_data/models'  #
    body_model = smplx.create(model_folder, model_type='smpl', gender='male', num_betas=10,
                                   batch_size=batch_size)  # we are running all the datasets for male gendern
    body_model = body_model.to(device=device)
    beta = np.zeros((10))
    betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0).repeat(batch_size, 1).to(device)
    # humor_dir = '/BS/humanpose/static00/experiments/humor/results/out/hps_denoise1/results_out'
    # gt_dir = '/BS/humanpose/static00/experiments/humor/results/out/hps_denoise1/results_out'
    hand_pose = torch.zeros((batch_size, 6)).to(device)
    gt_dir = '/BS/humanpose/static00/experiments/humor/results/out/{}_{}/results_out'.format(data_name, batch_size)
    humor_dir = '/BS/humanpose/static00/experiments/humor/results/out/{}_{}/results_out'.format(data_name, batch_size)
    vposer_root = '/BS/humanpose/static00/experiments/vposer/results/motion_exp/pose_experiments_results/{}_{}/optim'.format(
        data_name, batch_size)

    vposer_root = '/BS/humanpose/static00/experiments/vposer/{}_{}/optim'.format(
        data_name, batch_size)
    print(vposer_root)
    # ours_dir = '/BS/garvita4/static00/pose_manifold/amass/euc_data/Abs_False_Raw_dc/lrelu_l1_None_1e-06/ckpt_{}/{}_{}'.format(
    #     ckpt, data_name, batch_size)
    all_dirs = sorted(os.listdir(humor_dir))

    for dir_name in all_dirs:
        gt_data = os.path.join(gt_dir, dir_name, 'gt_results.npz')
        humor_results = os.path.join(humor_dir, dir_name, 'stage3_results.npz')
        observations = os.path.join(humor_dir, dir_name, 'observations.npz')
        vposer_optim_data = os.path.join(humor_dir, dir_name, 'stage2_results.npz')
        # vposer_optim_data = observations
        # vposer_proj_data = os.path.join(vposer_proj, dir_name ,'vposer_v1.npz')
        # if os.path.exists(our_data):
        #     ipdb.set_trace()
        # if os.path.exists(gt_data) and os.path.exists(humor_results) and os.path.exists(our_data) and  os.path.exists(observations) and   os.path.exists(vposer_optim_data) and  os.path.exists(vposer_proj_data):
        if os.path.exists(gt_data) and os.path.exists(humor_results) and os.path.exists(
                observations) and os.path.exists(vposer_optim_data):
            gt_pose = torch.from_numpy(np.load(gt_data)['pose_body']).float()
            humor_pose = torch.from_numpy(np.load(humor_results)['pose_body']).float()
            observations_pose = torch.from_numpy(np.load(observations)['pose_body']).float()
            vposer_optim_pose = torch.from_numpy(np.load(vposer_optim_data)['pose_body']).float()
            if torch.sum(torch.isnan(vposer_optim_pose)) >1:
                # print('nan values in vpose pose')
                continue
            smpl_posed_gt = body_model.forward(betas=betas_torch, body_pose=torch.cat((gt_pose, hand_pose), dim=1))
            smpl_posed_humor = body_model.forward(betas=betas_torch, body_pose=torch.cat((humor_pose, hand_pose), dim=1))
            smpl_posed_obs = body_model.forward(betas=betas_torch, body_pose=torch.cat((observations_pose, hand_pose), dim=1))

            smpl_posed_optim = body_model.forward(betas=betas_torch, body_pose=torch.cat((vposer_optim_pose, hand_pose), dim=1))
            err_humor = torch.mean(
                torch.sqrt(torch.sum((smpl_posed_humor.vertices - smpl_posed_gt.vertices) ** 2, dim=2)))
            err_obs = torch.mean(torch.sqrt(torch.sum((smpl_posed_obs.vertices - smpl_posed_gt.vertices) ** 2, dim=2)))
            err_optim = torch.mean(
                torch.sqrt(torch.sum((smpl_posed_optim.vertices - smpl_posed_gt.vertices) ** 2, dim=2)))
            # err_proj = torch.mean(torch.sqrt(torch.sum((smpl_posed_proj.vertices - smpl_posed_gt.vertices)**2, dim=2)))
            # print(dir_name, 'observation error: {:02f}  humor error: {:02f}  '.format(err_obs.item()*100, err_humor.item()*100))
            all_err_humor.append(err_humor.detach().numpy())
            all_err_obs.append(err_obs.detach().numpy())
            all_err_vposer.append(err_optim.detach().numpy())

            if rep == 'quat':
                observations_pose = axis_angle_to_quaternion(observations_pose.reshape(batch_size, 21, 3))
                gt_pose = axis_angle_to_quaternion(gt_pose.reshape(batch_size, 21, 3))
                vposer_optim_pose = axis_angle_to_quaternion(vposer_optim_pose.reshape(batch_size, 21, 3))
                humor_pose = axis_angle_to_quaternion(humor_pose.reshape(batch_size, 21, 3))

                #calculate noise from gt using SMPL vertices

                err_im = torch.mean( torch.sqrt(torch.sum((observations_pose - mean) ** 2, dim=2)))
                sd_var = torch.sqrt(torch.mean(((observations_pose - mean) ** 2)/var))
                all_sd_var.append(sd_var.detach().numpy())
                all_err_im.append(err_im.detach().numpy())
                # print(sd_var.detach().numpy(), all_err_obs[-1] * 100,  all_err_vposer[-1] * 100,  all_err_humor[-1] * 100 )
                #calculate deviation from mean
    # np.savez()

    #create bins
    all_sd_var = np.array(all_sd_var)
    all_err_obs = np.array(all_err_obs)
    all_err_vposer = np.array(all_err_vposer)
    all_err_humor = np.array(all_err_humor)
    print(np.max(all_sd_var))

    for i in range(11):
        sigma_1_idx = all_sd_var < i

        sigma_1_obs = np.mean(all_err_obs[sigma_1_idx])
        sigma_1_vposer = np.mean(all_err_vposer[sigma_1_idx])
        sigma_1_humor = np.mean(all_err_humor[sigma_1_idx])
        print(i)
        print(sigma_1_obs* 100, sigma_1_vposer* 100, sigma_1_humor* 100)


