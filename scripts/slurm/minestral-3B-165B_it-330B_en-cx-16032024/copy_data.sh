#!/bin/bash
#SBATCH --job-name=minestral-350m-7B_it-7B_en-cx-13022024    # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-cx-13022024/extract_for_tokenizer-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-cx-13022024/extract_for_tokenizer-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial # boost_usr_prod

# module load profile/deeplrn culturax/2309
# module load openmpi

# export OMP_PROC_BIND=true
# export OMP_NUM_THREADS=8
# export HF_DATASETS_CACHE=$SCRATCH/hf_cache

# source ~/llmfoundry-cuda-flash-attn2-env/bin/activate
# --data_folder $WORK/hplt_res/reservoir_sample_10M_80M/en/web \

# mpirun -n 1 --rank-by core -bind-to core 

# ~/llmfoundry-cuda-flash-attn2-env/bin/python \
#     /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/extract_txt_for_tokenizer.py \
#     --data_folder $WORK/culturax_res/reservoir_sample_10M_100M/en/web \
#     --file_to_save $SCRATCH/data/minestral-350m-7B_it-7B_en-cx-13022024/data/tokenizer/en.txt \
#     --max_samples 1_900_000

# ~/llmfoundry-cuda-flash-attn2-env/bin/python \
#     /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/extract_txt_for_tokenizer.py \
#     --data_folder $WORK/culturax_res/reservoir_sample_10M_100M/it/web \
#     --file_to_save $SCRATCH/data/minestral-350m-7B_it-7B_en-cx-13022024/data/tokenizer/it.txt \
#     --max_samples 1_900_000

rsync -azvp /leonardo_scratch/large/userexternal/rorland1/data/minestral-1B-100B_it-100B_en-cx-04032024/data/processed/it /leonardo_scratch/fast/IscrB_medit/data/minestral-1B-100B_it-100B_en-cx-04032024/data/processed