#!/bin/bash
#SBATCH --job-name=extract-redpj         # Job name
#SBATCH -o /leonardo_scratch/large/userexternal/rorland1/llm-foundry/training_logs/minerva-7B-900B_it-900B_en-200B-code-21052024/extract-redpj-2.out       # Name of stdout output file
#SBATCH -e /leonardo_scratch/large/userexternal/rorland1/llm-foundry/training_logs/minerva-7B-900B_it-900B_en-200B-code-21052024/extract-redpj-2.err       # Name of stderr error file

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial # boost_usr_prod
#SBATCH --time 4:00:00  
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8


module load profile/deeplrn

source /leonardo_scratch/large/userexternal/rorland1/python-envs/llm-foundry-0.8.0-venv/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

python scripts/misc/decompress_and_batch.py
