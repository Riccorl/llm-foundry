#!/bin/bash

# if NODES is 1, just run the training command, else use srun to run the training command on all nodes

# if [ "$NODES" -eq 1 ]; then
#     composer $TRAINING_SCRIPT $CONFIG_PATH
# else
#     srun $CURRENT_DIR/helpers/composer_cmd.sh
# fi

# if INTERACTIVE is set, run the training command in interactive mode
if [ "$INTERACTIVE" = "TRUE" ]; then
    composer $TRAINING_SCRIPT $CONFIG_PATH
else
    srun $CURRENT_DIR/helpers/composer_cmd.sh
fi
