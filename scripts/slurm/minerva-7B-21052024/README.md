# Minerva 7B

Training of Minerva 7B model.

Data Selection:
- TODO

## Download Fineweb

```bash

```

## Tokenizer Training

Extract the txt from datasets

```bash
# fineweb, example CC 2024
python scripts/data_prep/extract_txt_for_tokenizer.py \
    --dataset /leonardo_scratch/large/userexternal/rorland1/data/fineweb/data/CC-MAIN-2024-10/ \
    --output-file /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/fineweb.txt \
    --data-type parquet \
    --streaming \
    --max-samples 2000000
```

Train the tokenizer

Tokenizers Trainer (CulturaX, RedPajamaV2, FineWeb, StarCoder2)

```bash
bash scripts/slurm/base_scripts/train_tokenizer.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/fineweb.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.it.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/redpajama_it_3GB.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/starcoder_1GB.txt \
    --model-prefix /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer \
    --vocab-size 51200 \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/tokenizer-train-minerva7b-tokenizers-trainer \
    -t 08:00:00
```

Test

```bash
bash scripts/slurm/base_scripts/train_tokenizer.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.en.100k.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/starcoder_1GB.100k.txt \
    --model-prefix /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024/test/hfbpetrainer \
    --vocab-size 51200 \
    --cpu 32 \
    -j tokenizer-train-minerva7b-hfbpetrainer \
    -t 00:30:00
```

## Data Preprocessing

### Scripts

Scripts we used to preprocess the data:

#### Italian Data

RedPajamaV2 Head

```bash
# redpajama
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/redpajamav2/it/documents_deduped_by_CC_fineweb \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/redpajama-head \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-redpajama-head \
    --max-tokens 900_000_000_000 \
    --data-type json \
    --filter-by-domain incontri,escort \
    --file_pattern "*head*"
```

RedPajamaV2 Middle

```bash
# redpajama
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/redpajamav2/it/documents_deduped_by_CC_fineweb \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/redpajama-middle \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-redpajama-middle \
    --max-tokens 900_000_000_000 \
    --data-type json \
    --filter-by-domain incontri,escort \
    --file_pattern *middle* 
```

CulturaX

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo/prod/data/ai/culturax/2309/it/train \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/culturax-filtered/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024-again/data-preprocessing-culturax-filtered \
    --max-tokens 900_000_000_000 \
    --data-type arrow \
    --filter-by-domain incontri,escort \
    --batch-size 1024
```

Wikipedia

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/phuguetc/wikipedia_it \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/wikipedia_it/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-wikipedia_it \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

Books

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/phuguetc/books_it \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/books_it/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-books_it \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

EurLex

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/hq/it/eurlex_it/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/eurlex_it/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-eurlex_it \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

Gazzetta

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/hq/it/gazetta \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/gazetta/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-gazetta \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

#### English Data

FineWeb

```bash
# fineweb
# change CC to the desired CC split
export CC=CC-MAIN-2024-10
# export CC=CC-MAIN-2023-50
# export CC=CC-MAIN-2023-40
# export CC=CC-MAIN-2023-23
# export CC=CC-MAIN-2023-14
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/fineweb/data/$CC \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/fineweb/$CC \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-fineweb-$CC \
    --max-tokens 900_000_000_000 \
    --data-type parquet
```

Books

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/hq/en/books \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/books/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-books_it \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

Arxiv

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/phuguetc/arxiv/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/arxiv/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 16 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-arxiv \
    --max-tokens 900_000_000_000 \
    --data-type jsonl \
    --skip-dataloader
```

```bash
export PART=part1
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/rorland1/data/hq/en/arxiv/$PART \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/arxiv/$PART \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-arxiv-$PART \
    --max-tokens 900_000_000_000 \
    --data-type jsonl \
    --skip-dataloader
```

StackExchange

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/phuguetc/stackexchange/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/stackexchange/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-stackexchange \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

Wikipedia

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/phuguetc/wikipedia_en \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/wikipedia_en/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-wikipedia_en \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

Gutenberg

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/phuguetc/books_en \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/en/gutenberg_en/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 16 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-gutenberg_en \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

#### Code Data

StarCoder2 (Code)

```bash
# starcoder
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/phuguetc/starcoder2_smol_batches \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/code/starcoder2/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-starcoder2 \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

```bash
# starcoder
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/phuguetc/starcoder2_data_ricc \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/code/starcoder2_part2/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-starcoder2-part2 \
    --max-tokens 900_000_000_000 \
    --data-type jsonl
```

### Data statistics

Number of tokens in the processed data:

- Italian
  - RedPajamaV2 687952502784 tokens
    - Head  240824430592 tokens
    - Middle 447128072192 tokens
  - CulturaX 158201876480 tokens
  - Wikipedia 1265135616 tokens
  - Books 147017728 tokens
  - Gazzetta 1654013952 tokens
  - EurLex 1647013888 tokens

- English
  - FineWeb 1076406624256 tokens
    - CC-MAIN-2024-10: 163075538944 tokens
    - CC-MAIN-2023-50: 250647552000 tokens
    - CC-MAIN-2023-40: 222520823808 tokens
    - CC-MAIN-2023-23: 228337487872 tokens
    - CC-MAIN-2023-14: 211825221632 tokens
  - Books 27740004352 tokens
  - Arxiv 33231106048 tokens
    - Part 1 8394358784 tokens
    - Part 2 8291110912 tokens
    - Part 3 8328163328 tokens
    - Part 4 8217473024 tokens
  - StackExchange 22069268480 tokens
  - Wikipedia 5259501568 tokens
  - Gutenberg 6947893248 tokens

- Code
  - StarCoder2 74456072192 + 126298828800
    - Pere part 126298828800 tokens
    - Other part 74456072192 tokens

## Training

```bash
# train
bash /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/base_scripts/train.sh \
    -c /leonardo/home/userexternal/rorland1/llm-foundry/scripts/slurm/minerva-7B-21052024/minerva-7B.yaml \
    -g 4 \
    -n 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/train/train-1-test \
    -x \
    -t 01:00:00
```
