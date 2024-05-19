#!/bin/bash
#SBATCH --job-name=benchmark-1gpu-2node-fullshard            # Job name
#SBATCH -o /leonardo_scratch/large/userexternal/rorland1/llm-foundry/training_logs/benchmark-1gpu-2node-fullshard/benchmark-1gpu-2node-fullshard.out       # Name of stdout output file
#SBATCH -e /leonardo_scratch/large/userexternal/rorland1/llm-foundry/training_logs/benchmark-1gpu-2node-fullshard/benchmark-1gpu-2node-fullshard.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod
#SBATCH --time 4:00:00  
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# mpirun -np 2 -x NPROCS -x WORLD_SIZE=$(($SLURM_NNODES * $NPROCS)) -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) -x MASTER_PORT=11234 -x PATH -bind-to none -map-by node -mca pml ob1 -mca btl ^openib ./submit_composer.sh

srun ./scripts/slurm/benchmark/train_multinode.sh
