#!/bin/bash

python $DOWNLOAD_SCRIPT \
    --dataset $DATASET \
    --output-folder $OUTPUT \
    $ALLOW \
    $RESUME
