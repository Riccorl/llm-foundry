#!/bin/bash
#SBATCH --job-name=minestral-350M-35B_it-35B_en-cx-17032024-upload            # Job name
#SBATCH -o logs/minestral-350M-35B_it-35B_en-cx-17032024/upload-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-350M-35B_it-35B_en-cx-17032024/upload-job.err       # Name of stderr error file

#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial

module load profile/deeplrn culturax/2309
# module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8
module load zlib/1.3--gcc--12.2.0 cuda/12.1

source ~/llmfoundry-0.6.0/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

bash scripts/inference/upload_checkpoints_hf_hub.sh \
    /leonardo_scratch/large/userexternal/rorland1/llm-foundry/runs/minestral-350M-35B_it-35B_en-cx-17032024 \
    /leonardo_scratch/large/userexternal/rorland1/checkpoints/ \
    Minerva-350M
