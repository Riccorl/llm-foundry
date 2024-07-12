#!/bin/bash
#SBATCH --job-name=minestral-1B-100B_it-100B_en-cx-04032024-data-it  # Job name
#SBATCH -o logs/minestral-1B-100B_it-100B_en-cx-04032024/data_prep-it.out              # Name of stdout output file
#SBATCH -e logs/minestral-1B-100B_it-100B_en-cx-04032024/data_prep-it.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=32      # number of threads per task
#SBATCH --time 8:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_hf.py \
    --dataset /leonardo/prod/data/ai/culturax/2309/it \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-1B-100B_it-100B_en-cx-04032024/data/processed/it \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
    --num_workers 32
