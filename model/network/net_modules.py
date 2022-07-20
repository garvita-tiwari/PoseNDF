import ipdb
import torch
import torch.nn as nn
import numpy as np
from model.network.net_utils import PosEncoder, get_parent_mapping


class SDF(nn.Module):

    def __init__(self, opt_can,enc):
        super(SDF, self).__init__()
        self.num_neuron = opt_can['total_dim']
        self.num_layers = opt_can['num_layers']

        # x_freq =  opt_can['x_freq']

        self.input_dim =  opt_can['in_dim']
        if enc == 'Str':
            self.input_dim = opt_can['num_parts']*6

        self.layers = nn.ModuleList()
        self.pose_enc =  opt_can['ff_enc']

        #
        # if self.pose_enc:
        #     self.input_dim =  opt_can['in_dim'] +  opt_can['in_dim'] * 2 * x_freq
        #     self.x_enc = PosEncoder(x_freq, True)

        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 1))
        #
        # self.actvn = nn.LeakyReLU(0.1)
        # self.out_actvn = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=0)
        if opt_can['act'] == 'softplus':
            self.actvn = nn.Softplus(beta=opt_can['beta'])

        if opt_can['act'] == 'relu':
            self.actvn = nn.ReLU()

        if opt_can['act'] == 'lrelu':
            self.actvn = nn.LeakyReLU(0.1)

        if opt_can['act'] == 'None':
            self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = None
        #
        # if opt_can['out_act'] == 'ReLU':
        #     self.out_actvn = nn.ReLU()
        # if opt_can['out_act'] == 'LeakyReLU':
        #     self.out_actvn = nn.LeakyReLU(0.1)
    def forward(self, x,beta=None):
        batch_size = x.shape[0]
        num_pts = x.shape[1]
        if self.pose_enc:  #todo : check this
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = self.x_enc.encode(x)
            x = x.reshape(batch_size, num_pts, x.shape[1])
        for i in range(self.num_layers - 1):
            if i == 0:
                x_net = x
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net
        x_net = self.layers[-1](x_net)

        if self.out_actvn is not None:
            x_net = self.out_actvn(x_net)
        return x_net


class NDF(nn.Module):

    def __init__(self, opt_can):
        super(NDF, self).__init__()
        hidden_dim = opt_can['hidden_dim']
        input_dim = opt_can['in_dim']
        self.fc_0 = nn.Conv1d(input_dim, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)


        displacment = 0.0722
        displacements = []
        displacements.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacements.append(input)
        self.displacements = torch.Tensor(displacements).cuda()



    def forward(self, p):
        p = p.permute(0,2,1)
        #p = torch.cat([p + d for d in self.displacements], dim=2)
        net = self.actvn(self.fc_0(p))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)
        return out


class BoneMLP(nn.Module):
    def __init__(self, bone_dim, bone_feature_dim,parent=-1):
        super(BoneMLP, self).__init__()
        if parent ==-1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, bone_feature_dim),
            nn.ReLU()
        )

    def forward(self, bone_feat):

        return self.net(bone_feat)


class StructureEncoder(nn.Module):
    """copied from LEAP code(CVPR21, Marko et al)"""
    def __init__(self, opt, local_feature_size=6):
        super().__init__()

        self.bone_dim = 4  # 3x3 for pose and 1x3 for joint loc  #todo: change this encodibg for quaternion
        self.input_dim = self.bone_dim  # +1 for bone length
        self.parent_mapping = get_parent_mapping('smpl')

        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.net = nn.ModuleList([ BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i]) for i in range(self.num_joints) ])

    def get_out_dim(self):
        return self.out_dim

    @classmethod
    def from_cfg(cls, config):
        return cls(
            local_feature_size=config['local_feature_size'],
            parent_mapping=config['parent_mapping']
        )

    def forward(self, quat):
        """
        Args:
            pose: B x num_joints x 3 x 3
            rel_joints: B x num_joints x 3
        """
        B = quat.shape[0]* quat.shape[1]
        quat = quat.reshape(B, 21, 4)


        # B, K = rel_joints.shape[0], rel_joints.shape[1]
        # bone_lengths = torch.norm(rel_joints.squeeze(-1), dim=-1).view(B, K, 1)  # B x num_joints x 1
        #
        # bone_features = torch.cat((pose.contiguous().view(B, K, -1),
        #                            rel_joints.contiguous().view(B, K, -1)), dim=-1)
        #
        # root_bone_prior = self.proj_bone_prior(bone_features.contiguous().view(B, -1))  # B, bottleneck
        # root_bone_prior = self.proj_bone_prior(quat.contiguous().view(B, -1))  # B, bottleneck

        # fwd pass through the bone encoder
        features = [None] * self.num_joints
        # bone_transforms = torch.cat((bone_features, bone_lengths), dim=-1)

        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(quat[:, i, :])
            else:
                inp = torch.cat((quat[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
        features = torch.cat(features, dim=-1)  # B x f_len
        return features