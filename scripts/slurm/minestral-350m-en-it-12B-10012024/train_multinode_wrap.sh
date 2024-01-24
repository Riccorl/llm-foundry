#!/bin/bash
#SBATCH --job-name=minestral-350m-en-it-12B-10012024-train-multinode            # Job name
#SBATCH -o logs/minestral-350m-en-it-12B-10012024/train-multinode-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-350m-en-it-12B-10012024/train-multinode-job.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00  
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive          

master_port=11111
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# module load cuda gcc nvhpc openmpi nccl

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry/scripts

export NPROCS=4  # number of GPUs per node
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline

# mpirun -np 2 -x NPROCS -x WORLD_SIZE=$(($SLURM_NNODES * $NPROCS)) -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) -x MASTER_PORT=11234 -x PATH -bind-to none -map-by node -mca pml ob1 -mca btl ^openib ./submit_composer.sh

srun ./slurm/minestral-350m-en-it-12B-10012024/train_multinode.sh
