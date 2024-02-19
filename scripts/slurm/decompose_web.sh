#!/bin/bash
#SBATCH --job-name=extract_data_from_hf_reservoir_it    # Job name
#SBATCH -o extract_data_from_hf_it-job.out              # Name of stdout output file
#SBATCH -e extract_data_from_hf_it-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS
#SBATCH --mem 30GB

#SBATCH -A IscrB_medit

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache_lm

source ~/__Work/llmfoundry-cuda-flash-attn2-env/bin/activate

~/__Work/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/lmoroni0/__Work/llm-foundry/scripts/data_prep/decompose_web.py
