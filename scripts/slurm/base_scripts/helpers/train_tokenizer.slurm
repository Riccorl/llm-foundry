#!/bin/bash

echo "Starting training"
echo "Training script: $TRAINING_SCRIPT"
echo "Input: $INPUT"
echo "Model Prefix: $MODEL_PREFIX"
echo "Vocab Size: $VOCAB_SIZE"
echo "Num Threads: $NUM_THREADS"

python $TRAINING_SCRIPT --input $INPUT --model-prefix $MODEL_PREFIX --vocab-size  $VOCAB_SIZE --num-threads $NUM_THREADS
