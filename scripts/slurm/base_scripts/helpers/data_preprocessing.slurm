#!/bin/bash

python $PROCESSING_SCRIPT \
    --dataset $INPUT \
    --out_root $OUTPUT \
    --tokenizer $TOKENIZER \
    --concat_tokens $MAX_TOKENS \
    --split $SPLIT \ 
    --num_workers $NUM_WORKERS \
    --dataloader_batch_size $BATCH_SIZE \
    $SHUFFLE
