#!/bin/bash
#SBATCH --job-name=minestral-350m-7B_it-7B_en-hplt-09022024    # Job name
#SBATCH -o logs/minestral-350m-7B_it-7B_en-hplt-09022024/data_prep-en-eval-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-7B_it-7B_en-hplt-09022024/data_prep-en-en-eval-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --time 8:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8 zlib/1.2.13--gcc--11.3.0 cuda/11.8

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

# ~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
#     --path /leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web \
#     --data_files /leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/0.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/1.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/2.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/3.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/4.jsonl \
#     --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-hplt-09022024/data/processed/en/0-4-3B \
#     --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-hplt-09022024/tokenizer/minestral-350m-7B_it-7B_en-hplt-09022024-hf \
#     --max_tokens 3_000_000_000
# --data_files /leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/5.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/6.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/7.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/8.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/9.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/10.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/11.jsonl,/leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web/12.jsonl \

~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/convert_dataset_json.py \
    --path /leonardo_work/IscrB_medit/hplt_res/reservoir_sample_10M_53M/en/web \
    --data_files_pattern "(19[0-9]).jsonl" \
    --out_root /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-hplt-09022024/data/processed/en/eval \
    --split train --concat_tokens 2048 --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-hplt-09022024/tokenizer/minestral-350m-7B_it-7B_en-hplt-09022024-hf \
    --max_tokens 70_000

