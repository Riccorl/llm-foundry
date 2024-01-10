#!/bin/bash

export HF_DATASETS_CACHE=$WORK/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate

#   --name_or_path llm-celafaremo/minestral-350m-en-04012024-hf \
#   --name_or_path /leonardo_work/IscrB_medit/training/checkpoints/minestral-350m-25122023 \

# Generate responses to prompts
python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/inference/hf_generate.py \
  --name_or_path /leonardo_work/IscrB_medit/training/checkpoints/minestral-350m-en-04012024 \
  --max_new_tokens 256 \
  --prompts \
    "The answer to life, the universe, and happiness is" \
    "Here's a quick recipe for baking chocolate chip cookies: Start by"