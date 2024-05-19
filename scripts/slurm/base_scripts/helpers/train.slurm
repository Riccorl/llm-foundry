#!/bin/bash

# if NODES is 1, just run the training command, else use srun to run the training command on all nodes

# if [ "$NODES" -eq 1 ]; then
#     composer $TRAINING_SCRIPT $CONFIG_PATH
# else
#     srun $CURRENT_DIR/helpers/composer_cmd.sh
# fi

export NPROCS=$GPU_PER_NODE # number of GPUs per node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=11111
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

# singolo nodo, 4 gpu, con e senza
# export NCCL_IB_SL=1
# export UCX_IB_SL=1
# export NVSHMEM_IB_SL=1
# export NVSHMEM_DISABLE_NCCL=1

# if INTERACTIVE is set, run the training command in interactive mode
if [ "$INTERACTIVE" = "TRUE" ] || [ "$NODES" -eq 1 ]; then
    composer $TRAINING_SCRIPT $CONFIG_PATH
else
    srun $CURRENT_DIR/helpers/composer_cmd.sh
fi
