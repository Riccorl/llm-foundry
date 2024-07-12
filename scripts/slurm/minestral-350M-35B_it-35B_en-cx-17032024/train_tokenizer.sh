#!/bin/bash
#SBATCH --job-name=train_tokenizer-minestral-350m-7B_it-7B_en-cx-13022024    # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-cx-13022024/train_tokenizer-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-cx-13022024/train_tokenizer-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/train_tokenizer.py \
    --input $SCRATCH/data/minestral-350m-7B_it-7B_en-cx-13022024/data/tokenizer/en.txt,$SCRATCH/data/minestral-350m-7B_it-7B_en-cx-13022024/data/tokenizer/it.txt \
    --model_prefix $SCRATCH/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024
