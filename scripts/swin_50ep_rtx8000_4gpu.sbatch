#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=14                # uses 1 compute core per task
#SBATCH --time=23:59:58
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:4  ## To request specific GPU (v100 or rtx8000)
#SBATCH --job-name=swin_50ep_rtx_4g_b64_old
#SBATCH --output=swin_50ep_rtx_4g_b64_old.out

cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python setup.py install --cuda_ext
cd ../

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  moby_main.py \
--cfg configs/moby_swin_tiny.yaml --data-path imagenet --batch-size 64 --output rtx_4gpu --tag rtx_4gpu_50ep_lrdefault_b64
