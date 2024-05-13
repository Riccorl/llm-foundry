#!/bin/bash

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
# export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

source ~/llmfoundry-0.6.0/bin/activate

# get parent folder in input
FOLDER=$1
HF_FOLDER=$2
MODEL_NAME=$3

# check if both inputs are provided
if [ -z "$FOLDER" ] || [ -z "$HF_FOLDER" ]; then
  echo "Please provide the folder containing the composer checkpoint and the folder where the HF checkpoint will be saved"
  exit 1
fi

# check if the HF_FOLDER exists
if [ ! -d "$HF_FOLDER" ]; then
  echo "The folder `$HF_FOLDER` does not exist"
  exit 1
fi

# extract the model name from the folder
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=$(basename $FOLDER)
fi

CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
# SCRIPT_DIR=$CURRENT_DIR/../../inference

# upload latest
# remove old "latest" folder if it exists
if [ -d "$HF_FOLDER"/"$MODEL_NAME"-latest ]; then
    echo "Removing old $HF_FOLDER/$MODEL_NAME-latest folder"
    rm -r "$HF_FOLDER"/"$MODEL_NAME"-latest
fi
python $CURRENT_DIR/convert_composer_to_hf.py \
    --composer_path "$FOLDER"/latest-rank0.pt \
    --hf_output_path "$HF_FOLDER"/"$MODEL_NAME"-latest \
    --output_precision bf16 \
    --hf_repo_for_upload sapienzanlp/"$MODEL_NAME"

# now itereate over all the checkpoints but the latest
for FILE in "$FOLDER"/*; do
    # skip if latest in the name
    if [[ $FILE == *"latest"* ]]; then
        continue
    fi
    # extract the checkpoint filename
    CHECKPOINT_NAME=$(basename $FILE)
    # the checkpoint name has the form ep0-ba32000-rank0.pt
    # we need to extract the step number (e.g., 32000)
    STEP=$(echo $CHECKPOINT_NAME | cut -d '-' -f2 | cut -d 'r' -f1 | cut -d 'b' -f2 | cut -d 'a' -f2)

    # if hf_output_path already exists, skip
    if [ -d "$HF_FOLDER"/"$MODEL_NAME"-step"$STEP" ]; then
        echo "The folder $HF_FOLDER/$MODEL_NAME-step$STEP already exists. Skipping..."
        continue
    fi

    # upload the checkpoint
    python $CURRENT_DIR/convert_composer_to_hf.py \
        --composer_path "$FILE" \
        --hf_output_path "$HF_FOLDER"/"$MODEL_NAME"-step"$STEP" \
        --output_precision bf16 \
        --revision step"$STEP" \
        --hf_repo_for_upload sapienzanlp/"$MODEL_NAME"
done
