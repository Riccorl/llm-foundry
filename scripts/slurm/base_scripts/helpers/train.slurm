#!/bin/bash

# if NODES is 1, just run the training command, else use srun to run the training command on all nodes

if [ "$NODES" -eq 1 ]; then
    composer $TRAINING_SCRIPT $CONFIG_FILE
else
    srun $CURRENT_DIR/helpers/composer_cmd.sh
fi
