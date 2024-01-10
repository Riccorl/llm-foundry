#!/bin/bash
#SBATCH --job-name=minestral-350m-en-it-12B-10012024-data_prep-12B-en    # Job name
#SBATCH -o logs/minestral-350m-en-it-12B-10012024/data_prep-12B-en-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-en-it-12B-10012024/data_prep-12B-en-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 10:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
    --path $WORK/culturax/extracted/350M-model/en/train \
    --out_root $WORK/llm-foundry/culturax/350M-model-12B-en-it/processed/en \
    --split train --concat_tokens 2048 --tokenizer $WORK/tokenizer/trained/minestreal350m-en-it-hf \
     --max_tokens 6_000_000_000


# ~/llmfoundry-cuda-env/bin/python scripts/data_prep/convert_dataset_json.py \
#     --path $WORK/culturax/extracted/350M-model/it/train \
#     --out_root $WORK/llm-foundry/culturax/350M-model-12B-en-it/processed/it \
#     --split train --concat_tokens 2048 --tokenizer $WORK/tokenizer/trained/minestreal350m-en-it-hf \
#     --max_tokens 6_000_000_000
