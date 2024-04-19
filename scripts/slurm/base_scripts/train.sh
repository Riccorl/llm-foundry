#!/bin/bash

USAGE="Usage: $0 [OPTIONS]

This script is used to train a model with various parameters.

Options:
  -h                Show this help message and exit
  -c CONFIG_PATH    Path to the configuration file
  -v PYTHON_ENV     Python environment to use
  -n NODES          Number of nodes to use
  -l LOGS_PATH      Path to store logs
  -m MODULES        Modules to load
  -t TIME           Time for the job
  -a ACCOUNT        Account to use for the job
  -p PARTITION      Partition to use for the job
  -j JOB_NAME       Name of the job
  -o STD_OUT        Path to standard output file
  -e STD_ERR        Path to standard error file
  -x                Use exclusive node
  -t TRAINING_SCRIPT Path to the training script
  -g GPU_PER_NODE   Number of GPUs per node

Invalid options will show this help message and exit.
"

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hc:v:n:l:m:t:a:p:j:e:o:xt:g:" opt; do
    case $opt in
    h)
        printf "%s$USAGE" && exit 0
        ;;
    c)
        CONFIG_PATH="$OPTARG"
        ;;
    v)
        PYTHON_ENV="$OPTARG"
        ;;
    n)
        NODES="$OPTARG"
        ;;
    l)
        LOGS_PATH="$OPTARG"
        ;;
    m)
        MODULES="$OPTARG"
        ;;
    t)
        TIME="$OPTARG"
        ;;
    a)
        ACCOUNT="$OPTARG"
        ;;
    p)
        PARTITION="$OPTARG"
        ;;
    j)
        JOB_NAME="$OPTARG"
        ;;
    o)
        STD_OUT="$OPTARG"
        ;;
    e)
        STD_ERR="$OPTARG"
        ;;
    x)
        EXCLUSIVE="TRUE"
        ;;
    t)
        TRAINING_SCRIPT="$OPTARG"
        ;;
    g)
        GPU_PER_NODE="$OPTARG"
        ;;
    \?)
        echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
        ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "CONFIG_PATH is not set. To set it, use the -c flag" && exit 1
fi

if [ -z "$PYTHON_ENV" ]; then
    PYTHON_ENV=~/llmfoundry-0.6.0/bin/activate
fi

if [ -z "$NODES" ]; then
    NODES=1
fi

if [ -z "$LOGS_PATH" ]; then
    # default logs path is in SCRATCH folder
    LOGS_PATH="$SCRATCH/llm-foundry/training_logs"
    # if logs path does not exist, create it
    if [ ! -d "$LOGS_PATH" ]; then
        mkdir -p "$LOGS_PATH"
    fi
fi

if [ -z "$MODULES" ]; then
    MODULES="profile/deeplrn cuda/12.1"
fi

if [ -z "$TIME" ]; then
    TIME=24:00:00
fi

if [ -z "$ACCOUNT" ]; then
    ACCOUNT=IscrB_medit
fi

if [ -z "$PARTITION" ]; then
    PARTITION=boost_usr_prod
fi

if [ -z "$JOB_NAME" ]; then
    echo "JOB_NAME is not set. To set it, use the -j flag" && exit 1
fi

if [ -z "$STD_OUT" ]; then
    STD_OUT="$LOGS_PATH/$JOB_NAME.out"
fi

if [ -z "$STD_ERR" ]; then
    STD_ERR="$LOGS_PATH/$JOB_NAME.err"
fi

if [ -z "$EXCLUSIVE" ]; then
    EXCLUSIVE=""
else
    EXCLUSIVE="--exclusive"
fi

if [ -z "$GPU_PER_NODE" ]; then
    GPU_PER_NODE=1
fi

module load $MODULES
source "$PYTHON_ENV"

# cd to the directory where this script is located
CURRENT_DIR="$(dirname "$0")"
export CURRENT_DIR
cd $CURRENT_DIR

if [ -z "$TRAINING_SCRIPT" ]; then
    # get training script from project folder
    TRAINING_SCRIPT="$(pwd)/../../../train/train.py"
fi

# if NODES is 1, then we don't need all this shit'
# check if NODES = 1

if [ $NODES -gt 1 ]; then
    export NPROCS=4 # number of GPUs per node
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=11111
    export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))    
fi

# check if $SCRATCH/hf_cache exists
if [ ! -d "$SCRATCH/hf_cache" ]; then
    mkdir -p "$SCRATCH/hf_cache"
fi

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# training params
export TRAINING_SCRIPT
export CONFIG_PATH

# echo the params
echo "CURRENT_DIR: $CURRENT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "NODES: $NODES"
echo "GPU_PER_NODE: $GPU_PER_NODE"
echo "LOGS_PATH: $LOGS_PATH"
echo "MODULES: $MODULES"
echo "TIME: $TIME"
echo "ACCOUNT: $ACCOUNT"
echo "PARTITION: $PARTITION"
echo "JOB_NAME: $JOB_NAME"
echo "STD_OUT: $STD_OUT"
echo "STD_ERR: $STD_ERR"
echo "EXCLUSIVE: $EXCLUSIVE"
echo "TRAINING_SCRIPT: $TRAINING_SCRIPT"

# srun ./scripts/slurm/minestral-3B-165B_it-330B_en-cx-16032024/train_multinode.sh
sbatch -p $PARTITION \
    -A $ACCOUNT \
    --nodes=$NODES \
    --ntasks=$NODES \
    --time=$TIME \
    --job-name=$JOB_NAME \
    --output=$STD_OUT \
    --error=$STD_ERR \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gres=gpu:$GPU_PER_NODE \
    $EXCLUSIVE \
    ./train.slurm
