#!/bin/bash
#SBATCH --job-name=minestral-350m-7B_it-7B_it-hplt-13022024    # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-cx-13022024/data_prep-it-7B-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-cx-13022024/data_prep-it-7B-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=16       # number of threads per task
#SBATCH --time 8:00:00          # format: HH:MM:SS
#SBATCH --mem 128G              # memory per node

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

# --data_files_pattern "(3[1-9]|[4-9]\d|100).jsonl"
# --data_files_pattern "(\d|[12]\d|30).jsonl"

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
    --path /leonardo_work/IscrB_medit/culturax_res/reservoir_sample_10M_100M/it/web \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/data/processed/it/train \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
