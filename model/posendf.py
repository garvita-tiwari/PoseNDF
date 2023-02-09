from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
import ipdb
#todo: create a base trainer
from model.network.net_modules import  StructureEncoder, DFNet

import time
import pickle as pkl
from torch.autograd import grad


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class PoseNDF(torch.nn.Module):

    def __init__(self, opt):
        super(PoseNDF, self).__init__()

        self.device = opt['train']['device']

        # create all the models:
        # self.shape_model = ShapeNet().to(self.device)
        # self.pose_model = PoseNet().to(self.device)
        self.enc = None
        if opt['model']['StrEnc']['use']:
            self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['DFNet']).to(self.device)
        
        
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.loss = opt['train']['loss_type']
        self.batch_size= opt['train']['batch_size']


        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        
       
    def train(self, mode=True):
        super().train(mode)


    def compute_distance(self, rand_pose):
        """online data generation, not used"""
        rand_pose = rand_pose.unsqueeze(1).repeat(1, len(self.train_poses),1,1)
        train_pose = self.train_poses.unsqueeze(0).repeat( len(rand_pose),1, 1,1)
        dist = torch.sum(torch.arccos(torch.sum(rand_pose*train_pose,dim=3)),dim=2)/2.0  #ToDo: replace with weighted sum, why sqrt??, refer to eq2
        dist_vals = torch.mean(torch.topk(dist,k=5,dim=1,largest=False)[0],dim=1)
        return dist_vals

    def forward(self, pose, dist_gt=None, train=True ):
        pose = pose.to(device=self.device).reshape(-1,21,4)
        if train:
            dist_gt = dist_gt.to(device=self.device).reshape(-1)
        rand_pose_in = torch.nn.functional.normalize(pose.to(device=self.device),dim=1)
        

        if self.enc:
            rand_pose_in = self.enc(rand_pose_in)
        dist_pred = self.dfnet(rand_pose_in)
        if train:
            loss = self.loss_l1(dist_pred[:,0], dist_gt)
            # eikonal term loss
            grad_val = gradient(rand_pose_in, dist_pred)
            eikonal_loss =  ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
            return loss, {'dist': loss , 'eikonal': eikonal_loss}
        else:
            return {'dist_pred': dist_pred }
            


