#!/bin/bash

module load profile/deeplrn cuda/12.1

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# read Huggingface token from .env file
# export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")


source /leonardo_scratch/large/userexternal/rorland1/python-envs/llm-foundry-0.8.0-venv/bin/activate

# cd to the directory where this script is located
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "CURRENT_DIR: $CURRENT_DIR"
INFERENCE_DIR=$CURRENT_DIR/../../inference
cd $INFERENCE_DIR

# list of folders to upload

BASE_PATH=/leonardo_scratch/large/userexternal/rorland1/llm-foundry/runs/minestral-3B-165B_it-330B_en-cx-13042024

# LIST_OF_FOLDERS=(
#     "minestral-350M-35B_it-35B_en-cx-17032024-step10000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step12000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step16000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step18000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step2000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step20000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step22000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step24000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step26000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step28000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step30000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step32000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step34000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step35000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step36000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step37000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step38000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step39000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step4000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step40000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step41000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step42000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step43000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step44000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step45000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step46000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step47000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step47684"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step6000"
#     "minestral-350M-35B_it-35B_en-cx-17032024-step8000"
# )

# iterate over all the files in the directory
for CHECKPOINT_FILE in $BASE_PATH/*
do
    MODEL_NAME="sapienzanlp/Composer-Minerva-3B-base-v1.0"
    # file name is for instance ep0-ba96000-rank0.pt
    # extract the ba96000 part
    # STEPS=$(echo $CHECKPOINT_FILE | grep -o -P '(?<=ba)\d+' | head -1)
    if [[ $CHECKPOINT_FILE == *"latest"* ]]; then
        STEPS="latest"
    else
        STEPS=$(echo $CHECKPOINT_FILE | rev | cut -d'/' -f1 | rev | cut -d'-' -f2)
    fi
    # STEPS=$(echo $CHECKPOINT_FILE | rev | cut -d'/' -f1 | rev | cut -d'-' -f2)
    # FULL_MODEL_PATH=$BASE_PATH/$MODEL_PATH
    # if steps is latest set the steps to empty
    echo "MODEL_PATH: $CHECKPOINT_FILE"
    echo "MODEL_NAME: $MODEL_NAME"
    echo "REVISION: $STEPS"
    if [ "$STEPS" == "latest" ]; then
        bash $INFERENCE_DIR/upload_to_hf.sh $CHECKPOINT_FILE $MODEL_NAME
    else
        bash $INFERENCE_DIR/upload_to_hf.sh -r $STEPS $CHECKPOINT_FILE $MODEL_NAME
    fi
done

