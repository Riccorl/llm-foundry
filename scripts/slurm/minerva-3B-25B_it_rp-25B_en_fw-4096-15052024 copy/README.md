# Minerva 3B 4096 context length (15/05/2024)

Continual training of Minerva 3B model to extend the context length to 4096 tokens.

## Data Preprocessing

```bash

# redpajama
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo/prod/data/ai/culturax/2309/en \
    --output /leonardo_work/IscrB_medit/training/minestral-3B-165B_it-330B_en-cx-16032024-2048/data/processed/en \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
    --sequence-length 4096 \
    --max_tokens 330_000_000_000 \
    --shuffle \
    --cpu 32

# fineweb
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo/prod/data/ai/culturax/2309/en \
    --output /leonardo_scratch/large/userexternal/rorland1/data/training/minerva-3B-25B_it_rp-25B_en_fw-4096-15052024/data/processed/en \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minestral-350m-7B_it-7B_en-cx-13022024/tokenizer/minestral-350m-7B_it-7B_en-cx-13022024-hf \
    --sequence-length 4096 \
    --max_tokens 25_000_000_000 \
    --shuffle \
    --cpu 32
```

## Training

```bash
# train

```
