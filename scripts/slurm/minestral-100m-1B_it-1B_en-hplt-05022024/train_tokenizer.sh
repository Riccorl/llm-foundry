#!/bin/bash
#SBATCH --job-name=train_tokenizer-minestreal350m-en    # Job name
#SBATCH -o logs/minestral-100m-1B_it-1B_en-hplt-05022024/train_tokenizer-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-100m-1B_it-1B_en-hplt-05022024/train_tokenizer-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=32       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS
#SBATCH --mem 120G              # total memory in MB

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

# spm_train --input=culturax_100M_en_it.txt \
#     --model_prefix=minestreal350m-en-it \
#     --vocab_size=32768 \
#     --character_coverage=0.9999 \
#     --model_type=bpe \
#     --byte_fallback=true \
#     --split_digits true

# export OMP_PROC_BIND=true
# export HF_DATASETS_CACHE=$WORK/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

# ~/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/train_tokenizer.py \
#     --input "/leonardo_work/IscrB_medit/tokenizer/data/6B/en.txt,/leonardo_work/IscrB_medit/tokenizer/data/6B/it.txt" \
#     --model_prefix "/leonardo_work/IscrB_medit/tokenizer/trained/minestreal350m-en-it"

~/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/train_tokenizer.py \
    --input $SCRATCH/data/minestral-100m-1B_it-1B_en-hplt-05022024/data/tokenizer/en.txt,$SCRATCH/data/minestral-100m-1B_it-1B_en-hplt-05022024/data/tokenizer/it.txt \
    --model_prefix $SCRATCH/data/minestral-100m-1B_it-1B_en-hplt-05022024/tokenizer/minestral-100m-1B_it-1B_en-hplt-05022024
