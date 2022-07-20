"""Fit GMM to AMASS data"""
import sys
import time
from data_distribution import  PoseData
from data_splits import amass_splits

import torch

sys.path.append('/BS/garvita/work/code/libs/GMM-KMeans-PyTorch')

from gmm_pytorch import GMM_Batch
if __name__ == "__main__":

    amass_path = '/BS/humanpose/static00/data/PoseNDF_raw/smpl_h'
    modes = ['train', 'vald', 'test']
    mean = {}
    var = {}
    rep = 'rot6d'
    K = 8
    for mode in modes:
        dataloader_amass = PoseData(amass_path, amass_splits[mode], batch_size=1024)
        pose_data = torch.from_numpy(dataloader_amass.pose).cuda()
        st = time.time()
        gmm = GMM_Batch(K=K)
        _, pre_label = gmm.fit(pose_data, batch_size=1000, max_iters=200)
        pre_label = pre_label.detach().cpu().numpy()
        print(gmm.alpha)
        et = time.time()
        print(f"GMM-Batch-pytorch fitting time: {(et - st):.3f}ms")

        #
        # print(f"GMM-Batch-pytorch fitting time: {(et - st):.3f}ms")
        # ax = fig.add_subplot(1, 3, 3, projection='3d', facecolor='white')
        # ax.scatter(data[:n1, 0], data[:n1, 10], data[:n1, 20], c=pre_label[:n1])
        # ax.set_title(f"GMM-B:{(et - st):.1f}ms")
        #
        # plt.show()