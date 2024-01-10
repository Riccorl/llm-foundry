#!/bin/bash
#SBATCH --job-name=train_tokenizer-minestreal350m-en    # Job name
#SBATCH -o logs/minestral-350m-en-04012024/train_tokenizer-minestreal350m-en-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-en-04012024/train_tokenizer-minestreal350m-en-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=32       # number of threads per task
#SBATCH --time 4:30:00          # format: HH:MM:SS

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
    --input "/leonardo_work/IscrB_medit/tokenizer/data/10M/en.txt" \
    --model_prefix "/leonardo_work/IscrB_medit/tokenizer/trained/spm/minestreal350m-en-v2"
