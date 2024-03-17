#!/bin/bash
#SBATCH --job-name=minestral-1B-100B_it-100B_en-cx-04032024-train            # Job name
#SBATCH -o logs/minestral-1B-100B_it-100B_en-cx-04032024/train-test-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-1B-100B_it-100B_en-cx-04032024/train-test-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=4             # number of tasks per node
#SBATCH --cpus-per-task=8              # number of threads per task
#SBATCH --time 24:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:4                    # number of gpus per node
#SBATCH --exclusive                     # request nodes exclusively

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309 openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8
# module load 

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

composer scripts/train/train.py scripts/train/yamls/pretrain/minestral-1B-100B_it-100B_en-cx-04032024.yaml
