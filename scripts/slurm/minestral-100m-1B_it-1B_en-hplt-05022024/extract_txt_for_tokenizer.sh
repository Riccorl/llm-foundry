#!/bin/bash
#SBATCH --job-name=minestral-100m-1B_it-1B_en-hplt-05022024    # Job name
#SBATCH -o logs/minestral-100m-1B_it-1B_en-hplt-05022024/extract_for_tokenizer-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-100m-1B_it-1B_en-hplt-05022024/extract_for_tokenizer-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -p lrd_all_serial

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate
# --data_folder $WORK/hplt_res/reservoir_sample_10M_80M/en/web \


~/llmfoundry-cuda-flash-attn2-env/bin/python \
    /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/extract_txt_for_tokenizer.py \
    --data_folder $WORK/hplt_res/reservoir_sample_10M_80M/en/web \
    --file_to_save $SCRATCH/data/minestral-100m-1B_it-1B_en-hplt-05022024/data/tokenizer/en.txt \
    --max_samples 850_000

~/llmfoundry-cuda-flash-attn2-env/bin/python \
    /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/extract_txt_for_tokenizer.py \
    --data_folder $WORK/hplt_res/reservoir_sample_10M_80M/it/web \
    --file_to_save $SCRATCH/data/minestral-100m-1B_it-1B_en-hplt-05022024/data/tokenizer/it.txt \
    --max_samples 850_000
