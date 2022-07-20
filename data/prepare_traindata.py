""""Prepare training data"""

import os
import numpy as np
import pytorch3d
import faiss
import ipdb
from data_splits import amass_splits
import argparse
import torch
from create_data import PoseData, quat_doublecover, axis_angle_to_quaternion_np
import faiss
from pytorch3d.transforms import axis_angle_to_quaternion
import faiss.contrib.torch_utils
import dist_utils

def faiss_idx_np(amass_datas, root_dir):
    all_pose = []
    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))


        for seq in seqs:
            if not 'npz' in seq:
                continue
            # data = np.memmap(os.path.join(ds_dir, seq))
            # tmp+= data.size
            # print('cumulative....', tmp)
            data_val = np.load(os.path.join(ds_dir, seq))['pose_body']
            pose_pt = torch.from_numpy(data_val.reshape(len(data_val), 21, 3))
            data_val = axis_angle_to_quaternion(pose_pt).detach().numpy()
            # all_pose.extend(data_val.reshape(len(data_val), 84))
            data_val = quat_doublecover(data_val,samples=int(len(data_val)/10) )  #Todo: add a condition for double cover
            all_pose.extend(data_val.reshape(len(data_val), 84))
            print(seq)
    # print('total size....', tmp)
    data_all = np.array(all_pose)
    # index = faiss.index_factory(84, "Flat",faiss.METRIC_INNER_PRODUCT)
    index = faiss.index_factory(84, "Flat")
    index.train(data_all)
    index.add(data_all)
    return  index, data_all

def faiss_idx_torch(amass_datas, root_dir):
    all_pose = []
    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))

        for seq in seqs:
            if not 'npz' in seq:
                continue
            # data = np.memmap(os.path.join(ds_dir, seq))
            # tmp+= data.size
            # print('cumulative....', tmp)
            data_val = np.load(os.path.join(ds_dir, seq))['pose_body']
            pose_pt = torch.from_numpy(data_val.reshape(len(data_val), 21, 3))
            # data_val = axis_angle_to_quaternion(pose_pt).detach().numpy()
            data_val = axis_angle_to_quaternion(pose_pt)
            # Todo: add  double cover
            #change data to quaternion

            all_pose.extend(data_val.reshape(len(data_val), 84).unsqueeze(0))
    # print('total size....', tmp)
    print('total data szie......', len(all_pose))

    data_all = torch.cat(all_pose).to(device='cuda')
    res = faiss.StandardGpuResources()
    nlist = 10000   #Todo chnage this number, I think more is faster
    # index = faiss.GpuIndexIVFFlat(res, 84, nlist, faiss.METRIC_L2)
    index = faiss.GpuIndexFlatL2(res, 84)
    index.train(data_all)
    index.add(data_all)

    return  index,data_all
def main(args):
    device ='cuda'
    #read the pose data:
    seq_file = os.path.join(args.raw_data, args.seq_file)
    if not os.path.exists(seq_file):
        print('Missing file.....', seq_file)
        return
    if not os.path.exists(os.path.join(args.out_dir, args.seq_file.split('/')[0])):
        print('Missing folder.....', os.path.join(args.out_dir, args.seq_file.split('/')[0]))
        os.makedirs(os.path.join(args.out_dir, args.seq_file.split('/')[0]))

    out_dsdir = os.path.join(args.out_dir, args.seq_file.split('/')[0])
    if not os.path.exists(out_dsdir):
        os.makedirs(out_dsdir)
    #Create dataloader
    query_data = PoseData(seq_file,mode='query', batch_size=1,num_samples=args.num_samples )
    query_data_loader = query_data.get_loader()

    #create distance calcultor
    distance_calculator = getattr(dist_utils, args.metric)
    distance_calculator = distance_calculator(args.batch_size, device)

    #Find KNN using sequential and batch operation:
    amass_datas = sorted(amass_splits['train'])

    if args.data_type == 'np':
        faiss_model,data_all = faiss_idx_np(amass_datas, args.raw_data)
    else:
        faiss_model,data_all = faiss_idx_torch(amass_datas, args.raw_data)

    k_faiss = args.k_faiss
    k_dist = args.k_dist
    for query_batch in query_data_loader:
        quer_pose = query_batch.get('pose').to(device=device)[0]
        if args.data_type == 'np':
            inp_pose = quer_pose.reshape(len(quer_pose),84).detach().cpu().numpy()
        else:
            inp_pose= quer_pose.reshape(len(quer_pose), 84)

        #for every query pose fine knn using faiss and then calculate exact enighbous using custom distance
        distances, neighbors = faiss_model.search(inp_pose, k_faiss)

        nn_poses = data_all[neighbors].reshape(len(quer_pose), k_faiss, 21,4)
        if args.data_type == 'np':
            nn_poses =  torch.from_numpy(nn_poses).to(device=device)

        dist, nn_id = distance_calculator.dist_calc(quer_pose,nn_poses, k_faiss, k_dist)
        nn_id = nn_id.detach().cpu().numpy()
        nn_poses = nn_poses.detach().cpu().numpy()
        nn_pose = []
        for idx in range(len(quer_pose)):
            nn_pose.append(nn_poses[idx][nn_id[idx]])

        # distances, neighbors = faiss_model.search(quer_pose.reshape(len(quer_pose), 84).cpu().detach().numpy(), 100)
        np.savez(os.path.join(args.out_dir, args.seq_file), dist=dist.detach().cpu().numpy(),nn_pose =np.array(nn_pose), pose=quer_pose.detach().cpu().numpy())
        # for amass_data in amass_datas:
        #     ds_dir = os.path.join(args.raw_data,amass_data)
        #     seqs = sorted(os.listdir(ds_dir))
        #
        #     for seq in seqs:
        #         if not 'npz' in seq:
        #
        #             continue
        #         ref_file = os.path.join(ds_dir, seq)
        #         ref_data = PoseData(ref_file, mode='ref', batch_size=1, num_samples=args.num_samples)
        #         ref_data_loader = ref_data.get_loader()
        #
        #         for ref_batch in ref_data_loader:
        #             # print('running for....', ref_file)   #todo: why 2 times
        #             ref_pose = ref_batch.get('pose').to(device=device)[0]
        #             # print(ref_pose.shape)
        #             if max < len(ref_pose):
        #                 max = len(ref_pose)
        #                 print(max)
        #             # dist, nn_id = distance_calculator.dist_calc(quer_pose, ref_pose)
        #             # all_id.append(nn_id.unsqueeze(2))
        #             # all_dist.append(dist.unsqueeze(2))
        #             # all_files.append(seq_file)
    # print(max)
    # ipdb.set_trace()
    # # all_id = torch.cat(all_id)
    # all_dist = torch.cat(all_dist,dim=2)
    # all_dist = all_dist.reshape(len(all_dist),-1)
    # try:
    #
    #     geo_val, geo_idx = tor ch.topk(all_dist, k=5, largest=False)
    #     np.savez(os.path.join(args.out_dir, args.seq_file), dist=geo_val.detach().cpu().numpy(), idx=geo_idx.detach().cpu().numpy() )
    # except:
    #     np.savez(os.path.join(args.out_dir, args.seq_file.replace('.npz', '_tmp.npz')), dist=all_dist.detach().cpu().numpy(), idx=np.array(all_id) )
    #
    # #do index select

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blendify example 02: Render mesh with albedo and depth.")

    # Paths to output files
    parser.add_argument("-rd", "--raw_data", type=str, default="/BS/humanpose/static00/data/PoseNDF_raw/smpl_h",
                        help="Path to the resulting image")
    parser.add_argument("-od", "--out_dir", type=str,
                        default="/BS/humanpose/static00/data/PoseNDF_train/smpl_h",
                        help="Path to the resulting image")
    parser.add_argument("-m", "--metric", type=str,
                        default='euc',
                        help="metric used for calculating nn")
    parser.add_argument("-dp", "--data_type", type=str,
                        default='np',
                        help="metric used for calculating nn")
    parser.add_argument("-sf", "--seq_file", type=str,
                        default='BioMotionLab_NTroje/rub001.npz',
                        help="sequence file")
    # Rendering parameters
    parser.add_argument("-n", "--num_samples", default=10000, type=int,
                        help="Number of pose sampled from each sequence")
    parser.add_argument("-k", "--k_dist", default=5, type=int,

                        help="K nearest neighbour")
    parser.add_argument("-kf", "--k_faiss", default=500, type=int,

                        help="K nearest neighbour")
    parser.add_argument("-bs", "--batch_size", default=128, type=int,
                        help="K nearest neighbour")

    arguments = parser.parse_args()
    main(arguments)