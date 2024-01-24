#!/bin/bash
#SBATCH -A AI4AL_LLMFT
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00  
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4        
#SBATCH --mem=0          
#SBATCH --job-name=llmfoundry

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

master_port=11111
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

module load cuda gcc nvhpc openmpi nccl
cd /leonardo_work/AI4AL_LLMFT/gfiameni/llm-foundry

source llmfoundry-venv/bin/activate

cd scripts

export NPROCS=4
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

# mpirun -np 2 -x NPROCS -x WORLD_SIZE=$(($SLURM_NNODES * $NPROCS)) -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) -x MASTER_PORT=11234 -x PATH -bind-to none -map-by node -mca pml ob1 -mca btl ^openib ./submit_composer.sh

srun ./submit_composer.sh