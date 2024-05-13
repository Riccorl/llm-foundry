#!/bin/bash

while getopts ":hv:m:" opt; do
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
COMPOSER_CKPT_PATH=${POSITIONAL_ARGS[0]}
OUTPUT_PATH=${POSITIONAL_ARGS[1]}
# this is optional and might not exist
MODEL_NAME=${POSITIONAL_ARGS[2]}

if [ -z "$PYTHON_ENV" ]; then
    PYTHON_ENV=~/llmfoundry-0.6.0/bin/activate
fi

if [ -z "$MODULES" ]; then
    MODULES="profile/deeplrn cuda/12.1"
fi

module load $MODULES
source "$PYTHON_ENV"

# cd to the directory where this script is located
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "CURRENT_DIR: $CURRENT_DIR"
INFERENCE_DIR=$CURRENT_DIR/../../inference
cd $INFERENCE_DIR

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

echo "COMPOSER_CKPT_PATH: $COMPOSER_CKPT_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "MODEL_NAME: $MODEL_NAME"

bash $INFERENCE_DIR/upload_checkpoints_hf_hub.sh $COMPOSER_CKPT_PATH $OUTPUT_PATH $MODEL_NAME
    # /leonardo_scratch/large/userexternal/rorland1/llm-foundry/runs/minestral-3B-165B_it-330B_en-cx-13042024 \
    # /leonardo_scratch/large/userexternal/rorland1/checkpoints/
