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

cmd="composer -v --world_size $WORLD_SIZE --base_rank $base_rank -n $NPROCS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $SLURM_NODEID train/train.py train/yamls/pretrain/mpt-125m.yaml   data_local=my-copy-c4   train_loader.dataset.split=train_small   eval_loader.dataset.split=val_small   max_duration=100ba   eval_interval=0   save_folder=mpt-125m"

echo "COMMAND: $cmd"
echo "-----------------------------------"

eval "$cmd"

#composer [-h] [--version] [-n NPROC] [--stdout STDOUT] [--stderr STDERR] [-v] [-m] [-c] [--world_size WORLD_SIZE]
#                [--base_rank BASE_RANK] [--node_rank NODE_RANK] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
#                training_script ...
