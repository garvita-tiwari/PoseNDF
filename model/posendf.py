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
from models.network.net_modules import  ShapeNet, PoseNet, WeightNet
from models.network.net_utils import net_arch, old_class

import time
import pickle as pkl

class PoseNDF(torch.nn.Module):

    def __init__(self, opt):
        super(PoseNDF, self).__init__()

        self.device = opt['train']['device']

        # get model parameters based on garment layer and smpl resolution
        in_dim, hidden_layers, out_dim, layer_size = net_arch(opt['experiment'])
        self.num_neigh = opt['experiment']['num_neigh']
        self.layer_size = layer_size
        self.batch_size = opt['train']['batch_size']
        # create all the models:
        # self.shape_model = ShapeNet().to(self.device)
        # self.pose_model = PoseNet().to(self.device)
        self.weight_net = WeightNet(in_dim, hidden_layers, out_dim).to(self.device)
        
        
        #geo_weights = np.load(os.path.join(DATA_DIR, 'real_g5_geo_weights.npy'))  todo: do we need this???
        self.loss = opt['train']['loss_type']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        
       
    def train(self, mode=True):
        super().train(mode)


    def compute_distance(self, rand_pose):
        rand_pose = rand_pose.unsqueeze(1).repeat(1, len(self.train_poses),1,1)
        train_pose = self.train_poses.unsqueeze(0).repeat( len(rand_pose),1, 1,1)
        dist = torch.sum(torch.arccos(torch.sum(rand_pose*train_pose,dim=3)),dim=2)/2.0  #ToDo: replace with weighted sum, why sqrt??, refer to eq2
        dist_vals = torch.mean(torch.topk(dist,k=5,dim=1,largest=False)[0],dim=1)
        return dist_vals

    def forward(self, inputs ):
        pose = inputs['pose'].to(device=self.device)
        rand_pose = torch.nn.functional.normalize(pose.to(device=self.device),dim=2)
        dist = self.compute_distance(rand_pose)
        dist_pred = self.model_occ(rand_pose.reshape(self.batch_size,84))
        loss = self.loss_l1(dist_pred[:,0], dist)
        return loss, {'dist': loss }



