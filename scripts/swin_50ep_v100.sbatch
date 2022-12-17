#!/bin/bash 
#SBATCH --nodes=4                        # requests 3 compute servers
#SBATCH --ntasks-per-node=8              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=23:59:59
#SBATCH --mem=12GB
#SBATCH --gres=gpu:v100:2  ## To request specific GPU (v100 or rtx8000)
#SBATCH --job-name=swin_50ep_v100_2g_12gb_continue
#SBATCH --output=v100_50ep_2g_swin_12gb_continue.out

cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
cd ../


python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  moby_main_v100.py \
--cfg configs/moby_swin_tiny_50ep.yaml --data-path imagenet --batch-size 64 --output v100_2gpu --tag v100_2gpu_50ep_lrdefault_swin --resume ./v100_2gpu/moby__swin_tiny__patch4_window7_224__odpr02_tdpr0_cm099_ct02_queue4096_proj2_pred2_50ep/v100_2gpu_50ep_lrdefault_swin/checkpoint.pth
