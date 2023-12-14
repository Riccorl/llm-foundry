#!/bin/bash
#SBATCH --job-name=data_prep    # Job name
#SBATCH -o logs/data_prep-job.out              # Name of stdout output file
#SBATCH -e logs/data_prep-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit

module load autoload profile/deeplrn culturax/2309

export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache

source llmf/bin/activate

python scripts/data_prep/convert_dataset_json.py \
    --path $WORK/culturax/extracted/en/train/ \
    --out_root $WORK/llm-foundry/processed/en \
    --split train --concat_tokens 2048 --tokenizer meta-llama/Llama-2-7b-hf \
    --compression zstd --max_tokens 1_000_000_000
