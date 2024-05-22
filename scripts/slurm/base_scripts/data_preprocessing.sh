#!/bin/bash

USAGE="Usage: train.sh [OPTIONS]

This script is used to train a model with various parameters.

Options:
  -h --help                 Show this help message and exit
  --input INPUT             Path to the input dataset or the dataset name from the Huggingface Hub
  --output OUTPUT           Path to the output dataset
  --tokenizer TOKENIZER     Tokenizer to use
  --sequence-length SEQ_LEN Maximum sequence length
  --max-tokens MAX_TOKENS   Maximum number of tokens to preprocess
  --batch-size BATCH_SIZE   Batch size of the dataloader for preprocessing
  --split SPLIT             Split of the dataset
  --data-type DATA_TYPE     Type of the dataset
  --filter-by-domain DOMAIN Filter the dataset by domain
  --filter-by-length LENGTH Filter the dataset by length
  --skip-dataloader SKIP_DATALOADER Skip the dataloader
  -c --cpu CPUS_PER_TASK    Number of CPUs per task
  -s --shuffle SHUFFLE      Shuffle the dataset
  -v PYTHON_ENV             Python environment to use
  -l LOGS_PATH              Path to store logs
  -m MODULES                Modules to load
  -t --time TIME            Time for the job
  -a ACCOUNT                Account to use for the job
  -p PARTITION              Partition to use for the job
  -j JOB_NAME               Name of the job
  -o STD_OUT                Path to standard output file
  -e STD_ERR                Path to standard error file
  -x                        Use exclusive node
  -i INTERACTIVE            Run the job interactively

Invalid options will show this help message and exit.
"

# Transform long options to short ones
for arg in "$@"; do
    shift
    case "$arg" in
    '--help') set -- "$@" '-h' ;;
    '--input') set -- "$@" '-u' ;;
    '--output') set -- "$@" '-d' ;;
    '--tokenizer') set -- "$@" '-y' ;;
    '--sequence-length') set -- "$@" '-r' ;;
    '--max-tokens') set -- "$@" '-q' ;;
    '--batch-size') set -- "$@" '-b' ;;
    '--split') set -- "$@" '-w' ;;
    '--data-type') set -- "$@" '-z' ;;
    '--filter-by-domain') set -- "$@" '-f' ;;
    '--filter-by-length') set -- "$@" '-g' ;;
    '--skip-dataloader') set -- "$@" '-k' ;;
    '--cpu') set -- "$@" '-c' ;;
    '--shuffle') set -- "$@" '-s' ;;
    '--log') set -- "$@" '-l' ;;
    '--modules') set -- "$@" '-m' ;;
    '--time') set -- "$@" '-t' ;;
    '--account') set -- "$@" '-a' ;;
    '--partition') set -- "$@" '-p' ;;
    '--job-name') set -- "$@" '-j' ;;
    '--std-out') set -- "$@" '-o' ;;
    '--std-err') set -- "$@" '-e' ;;
    '--exclusive') set -- "$@" '-x' ;;
    '--interactive') set -- "$@" '-i' ;;
    *) set -- "$@" "$arg" ;;
    esac
done

# check for named params
#while [ $OPTIND -le "$#" ]; do
while getopts ":hu:d:y:r:q:b:w:c:sv:l:m:t:a:p:j:o:e:xiz:f:g:k" opt; do
    case $opt in
    h)
        printf "%s$USAGE" && exit 0
        ;;
    u)
        INPUT="$OPTARG"
        ;;
    d)
        OUTPUT="$OPTARG"
        ;;
    y)
        TOKENIZER="$OPTARG"
        ;;
    r)
        SEQ_LEN="$OPTARG"
        ;;
    q)
        MAX_TOKENS="$OPTARG"
        ;;
    b)
        BATCH_SIZE="$OPTARG"
        ;;
    w)
        SPLIT="$OPTARG"
        ;;
    f) 
        DOMAIN="$OPTARG"
        ;;
    g)
        LENGTH="$OPTARG"
        ;;
    k)
        SKIP_DATALOADER="TRUE"
        ;;
    c)
        CPUS_PER_TASK="$OPTARG"
        ;;
    s)
        SHUFFLE=""
        ;;
    v)
        PYTHON_ENV="$OPTARG"
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
    i)
        INTERACTIVE="TRUE"
        ;;
    z)
        DATA_TYPE="$OPTARG"
        ;;
    \?)
        echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
        ;;
    esac
done

if [ -z "$INPUT" ]; then
    # raise an error if INPUT is not set
    echo "Input dataset is not set. To set it, use the $(--input) flag" && exit 1
fi

if [ -z "$OUTPUT" ]; then
    # raise an error if OUTPUT is not set
    echo "Output dataset is not set. To set it, use the $(--output) flag" && exit 1
fi

if [ -z "$TOKENIZER" ]; then
    # raise an error if TOKENIZER is not set
    echo "Tokenizer is not set. To set it, use the $(--tokenizer) flag" && exit 1
fi

if [ -z "$SEQ_LEN" ]; then
    # raise an error if SEQ_LEN is not set
    echo "Sequence length is not set. To set it, use the $(--sequence-length) flag" && exit 1
fi

if [ -z "$DATA_TYPE" ]; then
    # raise an error if DATA_TYPE is not set
    echo "Data type is not set. To set it, use the $(--data-type) flag" && exit 1
fi

if [ -z "$MAX_TOKENS" ]; then
    MAX_TOKENS=""
else
    MAX_TOKENS="--max_tokens $MAX_TOKENS"
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE="512"
fi

if [ -z "$SPLIT" ]; then
    SPLIT="train"
fi

if [ -z "$DOMAIN" ]; then
    DOMAIN=""
else
    # split DOMAIN by comma
    DOMAIN=$(echo $DOMAIN | tr "," " ")
    DOMAIN="--filter-by-domain $DOMAIN"
fi

if [ -z "$LENGTH" ]; then
    LENGTH=""
else
    LENGTH="--filter-by-length $LENGTH"
fi

if [ -z "$SKIP_DATALOADER" ]; then
    SKIP_DATALOADER=""
else
    SKIP_DATALOADER="--skip-dataloader"
fi

if [ -z "$CPUS_PER_TASK" ]; then
    CPUS_PER_TASK=32
fi

if [ -z "$SHUFFLE" ]; then
    SHUFFLE="--shuffle"
fi

if [ -z "$PYTHON_ENV" ]; then
    PYTHON_ENV=/leonardo_scratch/large/userexternal/rorland1/python-envs/llm-foundry-0.8.0-venv/bin/activate
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
    # it not interactive, raise an error
    if [ "$INTERACTIVE" = "FALSE" ]; then
        echo "JOB_NAME is not set. To set it, use the -j flag" && exit 1
    fi
fi

if [ -z "$LOGS_PATH" ]; then
    # default logs path is in SCRATCH folder
    LOGS_PATH="$SCRATCH/llm-foundry/training_logs"
    # if logs path does not exist, create it
    if [ ! -d "$LOGS_PATH/$JOB_NAME" ]; then
        mkdir -p "$LOGS_PATH/$JOB_NAME"
    fi
fi

if [ -z "$STD_OUT" ]; then
    # extract the last part of the JOB_NAME if it is a file path
    JOB_NAME_FILE_NAME=$(basename $JOB_NAME)
    STD_OUT="$LOGS_PATH/$JOB_NAME/$JOB_NAME_FILE_NAME.out"
fi

if [ -z "$STD_ERR" ]; then
    # extract the last part of the JOB_NAME if it is a file path
    JOB_NAME_FILE_NAME=$(basename $JOB_NAME)
    STD_ERR="$LOGS_PATH/$JOB_NAME/$JOB_NAME_FILE_NAME.err"
fi

if [ -z "$EXCLUSIVE" ]; then
    EXCLUSIVE=""
else
    EXCLUSIVE="--exclusive"
fi

if [ -z "$INTERACTIVE" ]; then
    INTERACTIVE="FALSE"
fi

module load $MODULES
source "$PYTHON_ENV"

# get the absolute path of the current directory
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
export CURRENT_DIR
cd $CURRENT_DIR

if [ -z "$PROCESSING_SCRIPT" ]; then
    # get training script from project folder
    PROCESSING_SCRIPT="$(pwd)/../../data_prep/convert_dataset_hf.py"
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

# export params
export PYTHON_ENV
export INTERACTIVE
export PROCESSING_SCRIPT

# echo the params
echo "CURRENT_DIR: $CURRENT_DIR"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "PROCESSING_SCRIPT: $PROCESSING_SCRIPT"

# data params
export INPUT
export OUTPUT
export TOKENIZER
export SEQ_LEN
export MAX_TOKENS
export BATCH_SIZE
export SPLIT
export SHUFFLE
export DATA_TYPE
export DOMAIN
export LENGTH
export SKIP_DATALOADER
if [ $CPUS_PER_TASK -eq 1 ]; then
    NUM_WORKERS=0
else
    NUM_WORKERS="$CPUS_PER_TASK"
fi
export NUM_WORKERS
echo "INPUT: $INPUT"
echo "OUTPUT: $OUTPUT"
echo "TOKENIZER: $TOKENIZER"
echo "SEQ_LEN: $SEQ_LEN"
echo "MAX_TOKENS: $MAX_TOKENS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SPLIT: $SPLIT"
echo "DATA_TYPE: $DATA_TYPE"
echo "DOMAIN: $DOMAIN"
echo "LENGTH: $LENGTH"
echo "SHUFFLE: $SHUFFLE"
echo "SKIP_DATALOADER: $SKIP_DATALOADER"
echo "NUM_WORKERS: $NUM_WORKERS"

if [ "$INTERACTIVE" = "TRUE" ]; then
    echo "Running job interactively"
    bash ./helpers/data_preprocessing.slurm
else
    echo "CPUS_PER_TASK: $CPUS_PER_TASK"
    echo "LOGS_PATH: $LOGS_PATH"
    echo "MODULES: $MODULES"
    echo "TIME: $TIME"
    echo "ACCOUNT: $ACCOUNT"
    echo "PARTITION: $PARTITION"
    echo "JOB_NAME: $JOB_NAME"
    echo "STD_OUT: $STD_OUT"
    echo "STD_ERR: $STD_ERR"
    echo "EXCLUSIVE: $EXCLUSIVE"
    echo "Running job non-interactively"
    sbatch -p $PARTITION \
        -A $ACCOUNT \
        --time=$TIME \
        --job-name=$JOB_NAME \
        --output=$STD_OUT \
        --error=$STD_ERR \
        --ntasks-per-node=1 \
        --cpus-per-task=$CPUS_PER_TASK \
        $EXCLUSIVE \
        ./helpers/data_preprocessing.slurm
fi
