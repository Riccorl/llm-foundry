#!/bin/bash
#SBATCH --job-name=minestral-350m-en-it-12B-10012024-multinode-train            # Job name
#SBATCH -o logs/minestral-350m-en-it-12B-10012024-multinode/train-%A_%a-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-350m-en-it-12B-10012024-multinode/train-%A_%a-job.err       # Name of stderr error file
#SBATCH --nodes=2                      # number of nodes
#SBATCH --ntasks-per-node=4             # number of tasks per node
#SBATCH --cpus-per-task=8              # number of threads per task
#SBATCH --time 0:02:00                  # format: HH:MM:SS
#SBATCH --exclusive                     # no other jobs on the node

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache
export WANDB_MODE=offline

# env variables for multi-node composer training
# WORLD_SIZE is the total number of GPUs across all nodes -> SLURM_NNODES * --gres=gpu:
export WORLD_SIZE=8
# NODE_RANK is the rank of the current node -> SLURM_NODEID
# export NODE_RANK=$SLURM_NODEID
# MASTER_ADDR is the IP address of the master node -> SLURM_JOB_NODELIST
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# MASTER_PORT is the port that master node uses for communication -> 54321
export MASTER_PORT=54321

# source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

srun composer --world_size $WORLD_SIZE --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT scripts/train/train.py scripts/train/yamls/pretrain/minestral-350m-en-it.yaml
srun composer --world_size $WORLD_SIZE --node_rank 1 --master_addr $MASTER_ADDR --master_port $MASTER_PORT scripts/train/train.py scripts/train/yamls/pretrain/minestral-350m-en-it.yaml

# salloc -A IscrB_medit -p boost_usr_prod --time 02:00:00 -N 2 --ntasks-per-node=4 --gres=gpu:4 --exclusive
