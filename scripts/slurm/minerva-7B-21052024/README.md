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

v1 (Only CulturaX and StarCoder2)

```bash
bash scripts/slurm/base_scripts/train_tokenizer.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/it.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/starcoder_1GB.txt \
    --model-prefix /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024 \
    --vocab-size 51200 \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/tokenizer-train-minerva7b \
    -t 04:00:00
```

v2 (CulturaX, RedPajamaV2, FineWeb, StarCoder2)

```bash
bash scripts/slurm/base_scripts/train_tokenizer.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/fineweb.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.it.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/redpajama_it_3GB.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/starcoder_1GB.txt \
    --model-prefix /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-v3 \
    --vocab-size 51200 \
    --cpu 32 \
    -j tokenizer-train-minerva7b-hf \
    -t 04:00:00
```

v3 (CulturaX, RedPajamaV2, FineWeb, StarCoder2) - 10 tabs

```bash
bash scripts/slurm/base_scripts/train_tokenizer.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/fineweb.en.txt,/leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/data/culturax.it.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/redpajama_it_3GB.txt,/leonardo_scratch/large/userexternal/phuguetc/Minerva-7B_tokenizer/starcoder_1GB.txt \
    --model-prefix /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-v3 \
    --vocab-size 51200 \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/tokenizer-train-minerva7b-v3 \
    -t 08:00:00
```

v4 (CulturaX, RedPajamaV2, FineWeb, StarCoder2) - Tokenizers Trainer

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

RedPajamaV2

```bash
# redpajama
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/fineweb/data/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/redpajama/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-redpajama \
    --max-tokens 900_000_000_000 \
    --data-type parquet
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/books_it/ \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/eurlex_it/ \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/gazetta/ \
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
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input /leonardo_scratch/large/userexternal/rorland1/data/fineweb/data/$CC \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/fineweb/$CC \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/books/ \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/arxiv/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 16 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-arxiv \
    --max-tokens 900_000_000_000 \
    --data-type jsonl \
    --skip-dataloader
```

StackExchange

```bash
bash scripts/slurm/base_scripts/data_preprocessing.sh \
    --input  /leonardo_scratch/large/userexternal/phuguetc/stackexchange/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/stackexchange/ \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/wikipedia_en/ \
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
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/gutenberg_en/ \
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
    --input /leonardo_scratch/large/userexternal/rorland1/data/starcoder/data/ \
    --output /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/starcoder/ \
    --tokenizer /leonardo_scratch/large/userexternal/rorland1/data/minerva-7B-750B_it-800B_en-200B-code-21052024/tokenizer/minerva-7B-750B_it-800B_en-200B-code-21052024-tokenizers-trainer-hf \
    --sequence-length 4096 \
    --shuffle \
    --cpu 32 \
    -j minerva-7B-900B_it-900B_en-200B-code-21052024/data-preprocessing-starcoder \
    --max-tokens 900_000_000_000 \
    --data-type parquet
```

### Data statistics

Number of tokens in the processed data:

- RedPajamaV2 _ tokens:
- Books IT 18571702272 tokens
- Wikipedia IT 1274011648 tokens
- FineWeb 1079829446656 tokens:
  - CC-MAIN-2024-10: 163,566,833,664 tokens
  - CC-MAIN-2023-50: 251,415,752,704 tokens
  - CC-MAIN-2023-40: 223,201,665,024 tokens
  - CC-MAIN-2023-23: 229,115,260,928 tokens
  - CC-MAIN-2023-14: 212,529,934,336 tokens

## Training

```bash
# train

```
