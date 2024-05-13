#!/bin/sh

echo "-----------------------------------"
echo "HOSTNAME: $(hostname -f)"
echo "NPROCS: $NPROCS"
echo "NODEID: $SLURM_NODEID"
BASE_RANK=$(($SLURM_NODEID * $NPROCS))
echo "WORLD SIZE: $WORLD_SIZE"
echo "BASE RANK: $BASE_RANK"
echo "MASTER addr: $MASTER_ADDR"
echo "MASTER port: $MASTER_PORT"
echo "Training script: $TRAINING_SCRIPT"
echo "Config file: $CONFIG_FILE"

TRAINING_COMMAND="composer -v --world_size $WORLD_SIZE --BASE_RANK $BASE_RANK -n $NPROCS --master_addr $MASTER_ADDR --master_port \
    $MASTER_PORT --node_rank $SLURM_NODEID $TRAINING_SCRIPT $CONFIG_FILE"

echo "COMMAND: $TRAINING_COMMAND"
echo "-----------------------------------"

eval "$TRAINING_COMMAND"
