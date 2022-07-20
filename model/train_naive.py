"""source: https://github.com/jchibane/ndf"""
from __future__ import division
import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn
from model.network.net_utils import gradient
import ipdb
from model.network import  net_modules


class BaseTrainer(object):

    def __init__(self,  train_dataset, val_dataset, opt):
        self.device = opt['train']['device']
        self.enc_name = 'Raw'


        self.model_occ = getattr(net_modules, opt['model']['CanSDF']['name'])
        self.model_occ = self.model_occ(opt['model']['CanSDF'],self.enc_name).to(self.device)
        self.optimizer_occ = getattr(optim, opt['train']['optimizer'])
        self.optimizer_occ = self.optimizer_occ(self.model_occ.parameters(), opt['train']['optimizer_param'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_poses = train_dataset.pose.to(device=self.device)
        self.val_poses = val_dataset.pose.to(device=self.device)

        self.loss = opt['train']['loss_type']
        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()

        self.exp_name = 'Naive'

        self.exp_name = '{}_{}_{}_{}'.format(self.exp_name,  opt['model']['CanSDF']['act'],self.loss,  opt['train']['optimizer_param'])
        self.exp_path = '{}/{}/'.format( opt['experiment']['root_dir'],self.exp_name )
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary')


        self.val_min = None
        self.train_min = None

        self.batch_size=  opt['train']['batch_size']

        self.out_act = None

        self.loss_weight = {'man_loss': 0, 'dist': 1.0, 'eikonal': 0}

    def train_step(self,batch, ep=None):

        self.model_occ.train()
        self.optimizer_occ.zero_grad()

        if self.enc_name == 'Str':
            self.model_str.train()
            self.optimizer_str.zero_grad()
        loss, loss_dict = self.compute_loss(batch, ep)
        loss.backward()
        self.optimizer_occ.step()
        if self.enc_name == 'Str':

            self.optimizer_str.step()
        return loss.item(), loss_dict

    def compute_distance(self, rand_pose):
        rand_pose = rand_pose.unsqueeze(1).repeat(1, len(self.train_poses),1,1)
        train_pose = self.train_poses.unsqueeze(0).repeat( len(rand_pose),1, 1,1)
        dist = torch.sum(torch.arccos(torch.sum(rand_pose*train_pose,dim=3)),dim=2)/2.0  #ToDo: replace with weighted sum, why sqrt??, refer to eq2
        dist_vals = torch.mean(torch.topk(dist,k=5,dim=1,largest=False)[0],dim=1)
        return dist_vals

    def compute_loss(self,batch,ep=None):
        device = self.device

        rand_pose = torch.nn.functional.normalize(batch.get('pose').to(device=device),dim=2)

        dist = self.compute_distance(rand_pose)
        dist_pred = self.model_occ(rand_pose.reshape(self.batch_size,84))
        loss = self.loss_l1(dist_pred[:,0], dist)


        return loss, {'dist': loss }

    def train_model(self, epochs, eval=True):
        loss = 0
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            sum_loss = 0
            loss_terms = {'man_loss': 0, 'dist': 0, 'eikonal': 0}
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

            for batch in train_data_loader:
                loss, loss_dict = self.train_step(batch, epoch)
                for k in loss_dict.keys():
                    loss_terms[k] += self.loss_weight[k]*loss_dict[k].item()
                    print("Current loss: {} {}  ".format(k, loss_dict[k].item()))

                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)
            print("Current batch_loss: {} {}  ".format(epoch, batch_loss))

            for k in loss_dict.keys():
                loss_terms[k] = loss_dict[k]/ len(train_data_loader)
            if self.train_min is None:
                self.train_min = batch_loss
            if batch_loss < self.train_min:
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'train_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'train_min={}'.format(epoch), [epoch, batch_loss])


            if eval and epoch > 2000:
                val_loss = self.compute_val_loss(epoch)
                print('validation loss:   ', val_loss)
                if self.val_min is None:
                    self.val_min = val_loss

                if val_loss < self.val_min:
                    self.val_min = val_loss
                    self.save_checkpoint(epoch)
                    for path in glob(self.exp_path + 'val_min=*'):
                        os.remove(path)
                    np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, batch_loss])
                self.writer.add_scalar('val loss batch avg', val_loss, epoch)

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', batch_loss, epoch)
            for k in loss_dict.keys():
                self.writer.add_scalar('training loss {} avg'.format(k), loss_terms[k] , epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            if self.enc_name == 'Str':

                torch.save({'epoch':epoch, 'model_state_occ_dict': self.model_occ.state_dict(),
                            'optimizer_occ_state_dict': self.optimizer_occ.state_dict(),
                            'model_state_str_dict': self.model_str.state_dict(),
                            'optimizer_str_state_dict': self.optimizer_str.state_dict()
                            }, path,  _use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch':epoch, 'model_state_occ_dict': self.model_occ.state_dict(),
                            'optimizer_occ_state_dict': self.optimizer_occ.state_dict()
                            }, path,  _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model_occ.load_state_dict(checkpoint['model_state_occ_dict'])
        self.optimizer_occ.load_state_dict(checkpoint['optimizer_occ_state_dict'])
        if self.enc_name == 'Str':
            self.model_str.load_state_dict(checkpoint['model_state_str_dict'])
            self.optimizer_str.load_state_dict(checkpoint['optimizer_str_state_dict'])

        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self, ep):

        self.model_occ.eval()
        sum_val_loss = 0

        val_data_loader = self.val_dataset.get_loader()
        for batch in val_data_loader:
            loss, _= self.compute_loss(batch, ep)
            sum_val_loss += loss.item()
        return sum_val_loss /len(val_data_loader)
