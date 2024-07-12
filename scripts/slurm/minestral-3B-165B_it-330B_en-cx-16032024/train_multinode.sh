#!/bin/sh

echo "-----------------------------------"
echo "HOSTNAME: $(hostname -f)"
echo "NPROCS: $NPROCS"
echo "NODEID: $SLURM_NODEID"
base_rank=$(($SLURM_NODEID * $NPROCS))
echo "WORLD SIZE: $WORLD_SIZE"
echo "BASE RANK: $base_rank"
echo "MASTER addr: $MASTER_ADDR"
echo "MASTER port: $MASTER_PORT"

cmd="composer -v --world_size $WORLD_SIZE --base_rank $base_rank -n $NPROCS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $SLURM_NODEID /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0/scripts/train/train.py /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0/scripts/train/yamls/pretrain/minestral-3B-165B_it-330B_en-cx-16032024.yaml"

echo "COMMAND: $cmd"
echo "-----------------------------------"

eval "$cmd"
