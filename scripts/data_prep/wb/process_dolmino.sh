#!/bin/bash

# PRELIMINARIES
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate llm-foundry

# DCLM
echo "Processing DCLM"
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset dclm \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/dclm \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 23600000000 \
    --val_tokens 4720000 \
    --shuffle

# FLAN
echo "Processing FLAN"
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset flan \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/flan \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 8300000000 \
    --val_tokens 1660000 \
    --shuffle

# PES20
echo "Processing PES20"
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset pes2o \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/pes2o \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 2925000000 \
    --val_tokens 585000 \
    --shuffle

# Wikipedia
echo "Processing Wikipedia"
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset wiki \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/wiki \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 3555000000 \
    --val_tokens 711000 \
    --shuffle

# Stackexchange
echo "Processing Stackexchange"
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset stackexchange \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/stackexchange \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 1225000000 \
    --val_tokens 245000 \
    --shuffle

# Math
echo "Processing Math"
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset math \
    --out_root /home/riccar_orlando/data/dolmino-mix-1124-50B/processed/math \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 10400000000 \
    --val_tokens 2080000 \
    --shuffle
