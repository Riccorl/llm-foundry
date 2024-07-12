#!/bin/bash
#SBATCH --job-name=minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024-8train            # Job name
#SBATCH -o logs/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024/train-8job.out       # Name of stdout output file
#SBATCH -e logs/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024/train-8job.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00  
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive     
#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8
module load zlib/1.3--gcc--12.2.0 cuda/12.1

source ~/llmfoundry-0.6.0/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

composer scripts/train/train.py scripts/train/yamls/pretrain/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024.yaml
