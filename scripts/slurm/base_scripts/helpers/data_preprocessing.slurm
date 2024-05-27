#!/bin/bash

python $PROCESSING_SCRIPT --dataset $INPUT --out_root $OUTPUT --tokenizer $TOKENIZER --concat_tokens $SEQ_LEN --split $SPLIT --num_workers $NUM_WORKERS --dataloader_batch_size $BATCH_SIZE --data_type $DATA_TYPE $MAX_TOKENS $SHUFFLE $DOMAIN $LENGTH $SKIP_DATALOADER $FILE_PATTERN
