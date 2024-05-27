#!/bin/bash
module load profile/deeplrn cuda/12.1

source ~/llmfoundry-0.6.0/bin/activate

cd /leonardo/home/userexternal/rorland1/llm-foundry-0.6.0

# export HF_DATASETS_CACHE=$SCRATCH/hf_cache
# export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
# export WANDB_MODE=offline
# # read Huggingface token from .env file
# export HF_TOKEN=$(cat .envs | grep HF_TOKEN | cut -d '=' -f2)


# folder to sync
FOLDER=$1

if [ -z "$FOLDER" ]; then
    echo "Please provide a folder to sync"
    exit 1
fi

# check if it is a wandb folder or a run folder
if [[ $FOLDER == *"run"* ]]; then
    # sync run folder
    echo "Syncing run folder $FOLDER"
    wandb sync $FOLDER
else
    # iterate over all runs in the folder
    echo "Syncing all runs in folder $FOLDER"
    for run in $(ls $FOLDER); do
        # sync run folder
        echo "Syncing run folder $FOLDER/$run"
        wandb sync $FOLDER/$run
    done
fi
