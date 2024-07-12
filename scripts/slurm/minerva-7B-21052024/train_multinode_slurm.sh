#!/bin/bash
#SBATCH --job-name=minerva-7B         # Job name
#SBATCH -o logs/minerva-7B/train-32job5.out       # Name of stdout output file
#SBATCH -e logs/minerva-7B/train-32job5.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00  
#SBATCH -N 32
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive          

master_port=11111
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

module load profile/deeplrn cuda/12.1

source /leonardo_scratch/large/userexternal/rorland1/python-envs/llm-foundry-0.8.0-venv/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry

export NPROCS=4  # number of GPUs per node
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

export NCCL_ASYNC_ERROR_HANDLING=1

# export NCCL_IB_SL=1
# export UCX_IB_SL=1
# export NVSHMEM_IB_SL=1
# export NVSHMEM_DISABLE_NCCL=1

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

srun .scripts/slurm/minerva-7B-21052024/train_multinode.sh
