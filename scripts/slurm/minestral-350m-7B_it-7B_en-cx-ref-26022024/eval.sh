#!/bin/bash
#SBATCH --job-name=minestral-350m-7B_it-7B_en-cx-ref-22022024-eval            # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-cx-ref-22022024/eval-job.out       # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-cx-ref-22022024/eval-job.err       # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=8              # number of threads per task
#SBATCH --time 4:00:00                  # format: HH:MM:SS
#SBATCH --gres=gpu:1                    # number of gpus per node

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file

cd ~/llm-foundry/
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

cd ~/llm-foundry/scripts

composer eval/eval.py eval/yamls/hf_eval.yaml
