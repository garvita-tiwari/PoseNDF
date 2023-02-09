import numpy as np

import torch
import torch.nn as nn
import pickle

from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct

class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class. [Source:Humor]
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 model_type='smpl'):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.
        '''

        if model_type == 'smpl':
            self.bm = SMPL(bm_path, num_betas=num_betas, batch_size=batch_size)
            self.num_joints = SMPL.NUM_JOINTS

        self.model_type = model_type

    def forward(self, root_orient=None, pose_body=None, betas=None,return_dict=False,**kwargs):

        out_obj = self.bm(
                betas=betas,
                global_orient=root_orient,
                body_pose=pose_body,
                return_full_pose=True,
        )

        out = {
            'vertices' : out_obj.vertices,
            'faces' : self.bm.faces_tensor,
            'betas' : out_obj.betas,
            'Jtr' : out_obj.joints,
            'body_pose' : out_obj.body_pose,
            'full_pose' : out_obj.full_pose
        }

        if not return_dict:
            out = Struct(**out)

        return out
