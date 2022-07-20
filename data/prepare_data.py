import argparse
import os

import numpy as np
import trimesh
#

from data_splits import amass_splits

def main(args):
    bash_path = args.bash_file
    amass_datas = amass_splits['train']
    with open(bash_path, 'w+') as fp:
        fp.write("#!/bin/bash"+ "\n")
        fp.write("#SBATCH -p gpu20"+ "\n")
        fp.write("#SBATCH -t 48:00:00"+ "\n")
        fp.write('#SBATCH -o "/scratch/inf0/user/gtiwari/slurm-%A.out"'+ "\n")
        fp.write('#SBATCH -e "/scratch/inf0/user/gtiwari/slurm-%A.err"'+ "\n")
        fp.write("#SBATCH --gres gpu:1"+ "\n")
        fp.write("source /BS/garvita/static00/software/miniconda3/etc/profile.d/conda.sh"+ "\n")
        fp.write("conda activate pytorch3d"+ "\n")
        count = 1
        for amass_data in amass_datas:
            ds_dir = os.path.join(args.raw_data,amass_data)
            seqs = sorted(os.listdir(ds_dir))

            for seq in seqs:
                if not 'npz' in seq:
                    continue
                fp.write("\t {})".format(count) + "\n")
                fp.write("\t\t\t")
                fp.write(
                    "python prepare_traindata.py --seq_file {}/{}".format(amass_data, seq) +
                     "& \n")
                count += 1
                fp.write("\t\t\t;;\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blendify example 02: Render mesh with albedo and depth.")

    # Paths to output files
    parser.add_argument("-rd", "--raw_data", type=str, default="/BS/humanpose/static00/data/PoseNDF_raw/smpl_h",
                        help="Path to the resulting image")
    parser.add_argument("-op", "--out_path", type=str, default="/BS/humanpose/static00/experiments/humor/results/out/amass_noise_0.5_60/results_vis",
                        help="Path to the resulting image")
    parser.add_argument("-bf", "--bash_file", type=str, default="./renderings.sh",
                        help="Path to the bash script file")
    parser.add_argument("-df", "--data_folder", type=str,
                        default="/BS/humanpose/static00/experiments/humor/results/out/amass_noise_0.5_60/results_out",
                        help="Path to the resulting image")


    arguments = parser.parse_args()
    main(arguments)
