#!/bin/bash
#SBATCH --job-name=minestral-350M-35B_it-35B_en-cx-17032024-2train-2        # Job name
#SBATCH -o logs/minestral-350M-35B_it-35B_en-cx-17032024/train-2job2.out       # Name of stdout output file
#SBATCH -e logs/minestral-350M-35B_it-35B_en-cx-17032024/train-2job2.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00  
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive          

master_port=11111
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

module load profile/deeplrn culturax/2309
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8
module load zlib/1.3--gcc--12.2.0 cuda/12.1

source ~/llmfoundry-0.6.0/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0

export NPROCS=4  # number of GPUs per node
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

# mpirun -np 2 -x NPROCS -x WORLD_SIZE=$(($SLURM_NNODES * $NPROCS)) -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) -x MASTER_PORT=11234 -x PATH -bind-to none -map-by node -mca pml ob1 -mca btl ^openib ./submit_composer.sh

srun ./scripts/slurm/minestral-350M-35B_it-35B_en-cx-17032024/train_multinode.sh