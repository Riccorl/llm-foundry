#!/bin/bash

module load profile/deeplrn cuda/12.1

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)

source ~/llmfoundry-0.6.0/bin/activate

# cd to the directory where this script is located
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "CURRENT_DIR: $CURRENT_DIR"
INFERENCE_DIR=$CURRENT_DIR/../../inference
cd $INFERENCE_DIR

# list of folders to upload

BASE_PATH=/leonardo_scratch/large/userexternal/rorland1/checkpoints

LIST_OF_FOLDERS=(
    "minestral-1B-100B_it-100B_en-cx-04032024-step10000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step12000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step16000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step18000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step2000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step20000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step22000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step24000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step26000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step28000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step30000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step32000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step34000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step35000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step36000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step37000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step38000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step39000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step4000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step40000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step41000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step42000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step43000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step44000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step45000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step46000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step47000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step47684"
    "minestral-1B-100B_it-100B_en-cx-04032024-step6000"
    "minestral-1B-100B_it-100B_en-cx-04032024-step8000"
)

# iterate over the list of folders
for MODEL_PATH in "${LIST_OF_FOLDERS[@]}"
do
    MODEL_NAME="sapienzanlp/Minerva-1B-base-v1.0"
    STEPS=$(echo $MODEL_PATH | cut -d'-' -f7)
    FULL_MODEL_PATH=$BASE_PATH/$MODEL_PATH
    # if steps is latest set the steps to empty
    echo "MODEL_PATH: $MODEL_PATH"
    echo "MODEL_NAME: $MODEL_NAME"
    echo "REVISION: $STEPS"
    if [ "$STEPS" == "latest" ]; then
        bash $INFERENCE_DIR/upload_to_hf.sh $FULL_MODEL_PATH $MODEL_NAME
    else
        bash $INFERENCE_DIR/upload_to_hf.sh -r $STEPS $FULL_MODEL_PATH $MODEL_NAME
    fi
done

