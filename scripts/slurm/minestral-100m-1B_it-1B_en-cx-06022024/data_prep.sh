#!/bin/bash
#SBATCH --job-name=minestral-100m-1B_it-1B_en-cx-06022024    # Job name
#SBATCH -o logs/minestral-100m-1B_it-1B_en-cx-06022024/data_prep-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-100m-1B_it-1B_en-cx-06022024/data_prep-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 8:00:00          # format: HH:MM:SS
#SBATCH --mem 128G              # memory per node

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
    --path /leonardo_work/IscrB_medit/culturax_res/reservoir_sample_10M_100M/en/web \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-100m-1B_it-1B_en-cx-06022024/data/processed/en \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-100m-1B_it-1B_en-cx-06022024/tokenizer/minestral-100m-1B_it-1B_en-cx-06022024-hf \
     --max_tokens 1_000_000_000

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
    --path /leonardo_work/IscrB_medit/culturax_res/reservoir_sample_10M_100M/it/web \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-100m-1B_it-1B_en-cx-06022024/data/processed/it \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-100m-1B_it-1B_en-cx-06022024/tokenizer/minestral-100m-1B_it-1B_en-cx-06022024-hf \
     --max_tokens 1_000_000_000
