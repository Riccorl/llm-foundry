#!/bin/bash

while getopts ":hv:m:o:c:a:t:r:" opt; do
    case $opt in
    h)
        printf "%s$USAGE" && exit 0
        ;;
    v)
        PYTHON_ENV="$OPTARG"
        ;;
    m)
        MODULES="$OPTARG"
        ;;
    o)
        ORGANIZATION="$OPTARG"
        ;;
    c)
        COMMIT_MESSAGE="$OPTARG"
        ;;
    a)
        ARCHIVE="$OPTARG"
        ;;
    t)
        REPO_TYPE="$OPTARG"
        ;;
    r)
        REVISION="$OPTARG"
        ;;
    \?)
        echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
        ;;
    esac
done

# shift for overrides
shift $((OPTIND - 1))
# split positional args
POSITIONAL_ARGS=$(echo "$@" | sed -e 's/ /\n/g')
# there are at most three positional args, and at least two
# first is the cmposer checkpoint path
# second is the output path
# third is the model name if it exists
POSITIONAL_ARGS=($POSITIONAL_ARGS)
MODEL_DIR=${POSITIONAL_ARGS[0]}
MODEL_NAME=${POSITIONAL_ARGS[1]}

# ensure that positional args are not empty
if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_NAME" ]; then
    echo "Model directory and model name must be provided" && echo "$USAGE" && exit 0
fi

if [ -z "$PYTHON_ENV" ]; then
    PYTHON_ENV=~/llmfoundry-0.6.0/bin/activate
fi

if [ -z "$MODULES" ]; then
    MODULES="profile/deeplrn cuda/12.1"
fi

if [ "$ORGANIZATION" != "" ]; then
    ORGANIZATION="--organization $ORGANIZATION"
fi

if [ "$COMMIT_MESSAGE" != "" ]; then
    COMMIT_MESSAGE="--commit-message $COMMIT_MESSAGE"
fi

if [ "$ARCHIVE" != "" ]; then
    ARCHIVE="--archive"
fi

if [ "$REPO_TYPE" != "" ]; then
    REPO_TYPE="--repo_type $REPO_TYPE"
else
    REPO_TYPE="--repo_type model"
fi

if [ "$REVISION" != "" ]; then
    REVISION="--revision $REVISION"
fi

module load $MODULES
source "$PYTHON_ENV"

# cd to the directory where this script is located
# CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
# echo "CURRENT_DIR: $CURRENT_DIR"
# INFERENCE_DIR=$CURRENT_DIR/../../inference
# cd $INFERENCE_DIR
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCE_DIR=$CURRENT_DIR

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

echo "MODEL_DIR: $MODEL_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "REVISION: $REVISION"

python $INFERENCE_DIR/upload_hf.py $MODEL_DIR \
    $MODEL_NAME \
    $ORGANIZATION \
    $COMMIT_MESSAGE \
    $ARCHIVE \
    $REPO_TYPE \
    $REVISION
