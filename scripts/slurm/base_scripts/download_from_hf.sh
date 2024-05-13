#!/bin/bash
#SBATCH --job-name=download_fineweb    # Job name
#SBATCH -o logs/download_fineweb/download_fineweb-job.out              # Name of stdout output file
#SBATCH -e logs/download_fineweb/download_fineweb-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 04:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial # boost_usr_prod

module load profile/deeplrn

export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export WANDB_MODE=offline
# get Huggingface token from python
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

source ~/llmfoundry-0.6.0/bin/activate

python /leonardo/home/userexternal/rorland1/llm-foundry-merge/scripts/data_prep/convert_dataset_hf.py
