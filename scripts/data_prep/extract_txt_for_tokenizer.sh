#!/bin/bash

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

python \
    scripts/data_prep/extract_txt_for_tokenizer.py \
    --data_folder /leonardo_work/IscrB_medit/culturax/extracted/350M-model/en/train/ \
    --file_to_save /leonardo_work/IscrB_medit/tokenizer/data/10Mfile_to_save/en.txt \
    --max_samples 10_000_000
