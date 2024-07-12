#!/bin/bash
#SBATCH --job-name=minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024-split    # Job name
#SBATCH -o logs/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024/split-4job.out              # Name of stdout output file
#SBATCH -e logs/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024/split-4job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=1       # number of threads per task
#SBATCH --time 4:00:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit
#SBATCH -p lrd_all_serial # boost_usr_prod

module load profile/deeplrn culturax/2309
module load openmpi

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

~/llmfoundry-cuda-flash-attn2-env/bin/python \
    scripts/slurm/minestral-1B-100B_it-100B_en-cx-continualhq10B-15032024/split_large_docs.py \
    04