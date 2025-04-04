#!/bin/bash

# PRELIMINARIES
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate llm-foundry

OUTPUT_FOLDER="/home/riccar_orlando/data/pdfs/semantic-scholar-parsed-jsonl-en-processed"

echo "Processing PDFS"
python scripts/data_prep/convert_dataset_hf.py --dataset /home/riccar_orlando/data/pdfs/semantic-scholar-parsed-jsonl-en \
    --out_root $OUTPUT_FOLDER \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 10_000_000_000 \
    --val_tokens 10_000_000 \
    --shuffle
