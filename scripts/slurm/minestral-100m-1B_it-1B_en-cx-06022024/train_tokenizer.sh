#!/bin/bash
#SBATCH --job-name=minestral-100m-1B_it-1B_en-cx-06022024-tokenizer    # Job name
#SBATCH -o logs/minestral-100m-1B_it-1B_en-cx-06022024/train_tokenizer-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-100m-1B_it-1B_en-cx-06022024/train_tokenizer-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=32       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS
#SBATCH --mem 120G              # total memory in MB

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/train_tokenizer.py \
    --input $SCRATCH/data/minestral-100m-1B_it-1B_en-cx-06022024/data/tokenizer/en.txt,$SCRATCH/data/minestral-100m-1B_it-1B_en-cx-06022024/data/tokenizer/it.txt \
    --model_prefix $SCRATCH/data/minestral-100m-1B_it-1B_en-cx-06022024/tokenizer/minestral-100m-1B_it-1B_en-cx-06022024
