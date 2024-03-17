#!/bin/bash
#SBATCH --job-name=minestral-350m-7B_it-7B_en-cx-ref-26022024    # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-cx-ref-26022024/debug.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-cx-ref-26022024/debug.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=16      # number of threads per task
#SBATCH --time 8:00:00          # format: HH:MM:SS
#SBATCH --mem 128G              # memory per node

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial # boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

# --data_files_pattern "(3[1-9]|[4-9]\d|100).jsonl"
# --data_files_pattern "(\d|[12]\d|30).jsonl"

# ~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/slurm/minestral-350m-7B_it-7B_en-cx-ref-22022024/filter_refs.py \
#     --path /leonardo_scratch/large/userexternal/lmoroni0/datasets/culturax_res/sequential/en/ \
#     --out_root /leonardo_scratch/large/userexternal/rorland1/data/culturax/sequential/web-refs/en

# ~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
#     --path /leonardo_scratch/large/userexternal/rorland1/data/culturax/sequential/web-refs/en \
#     --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-ref-26022024/data/processed/en/train \
#     --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
#     --max_tokens 7_000_000_000

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_hf.py \
    --dataset /leonardo/prod/data/ai/culturax/2309/it \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/debug/data/processed/it \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
    --max_tokens 7_000_000_000 --num_workers 32
