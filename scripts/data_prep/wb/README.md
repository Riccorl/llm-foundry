# LLM Continual Pretraining with LLM-Foundry

This document serves as a quick reference for the scripts and tools used to continue pretraining large language models (LLMs) with LLM-Foundry.

Ensure that you are running all commands from the `llm-foundry` directory and that your environment has all necessary dependencies installed.

This document is a work in progress and will be updated as the project evolves to include new features, tools, and best practices.

## Installation

```bash
git clone 
cd llm-foundry
pip install -e ".[gpu]"
```

## Data Processing

In this section, we will describe how to process datasets for use with LLM-Foundry.
The relevant scripts are located in the `scripts/data_prep` folder.
We will use the `convert_dataset_hf.py` script to convert datasets from Hugging Face into the LLM-Foundry format.

The script has been modified to work with any dataset from Hugging Face, as well as local files in JSON or JSONL format. 
However, each sample must be structured as a dictionary containing at least a `text` key, as shown below:

```json
{
    "text": "This is a sample text."
}
```

For wrapping local plain text files (e.g., parsed PDFs) into the required JSONL format, you can refer to the example code provided in `scripts/data_prep/wrap_to_jsonl.py`.

The output of the `convert_dataset_hf.py` script is a set of files in the [MDS](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/dataset_format.html#mds) format, which is optimized for efficient streaming and training.

### Dolmino

For our experiments, we will use the [Dolmino Mix](https://huggingface.co/datasets/allenai/dolmino-mix-1124) as our dataset for counter catastrophic forgetting.
The Dolmino Mix is a mix of several datasets, including DCLM (high-quality web bata), Flan (instructions), PES2O (academic papers), Wiki (Wikipedia and Wikidata), Stackexchange (Q&A), and Math.

We plan to use at most 50B tokens. Below are the proportions for each dataset, as described in the [OLMo 2](https://arxiv.org/abs/2501.00656) paper:

| Dataset       | Proportion |
| ------------- | ---------- |
| DCLM          | 0.472      |
| Flan          | 0.166      |
| PES2O         | 0.0585     |
| Wiki          | 0.0711     |
| Stackexchange | 0.0245     |
| Math          | 0.208      |

The following commands (which can be executed in sequence using the `process_dolmino.sh` script) will process the Dolmino Mix dataset.

Arguments explained:

- `--dataset`: The dataset to process. In this case, we will use the Dolmino Mix dataset.
- `--data_subset`: The subset of the dataset to process. This can be any of the datasets in the Dolmino Mix.
- `--out_root`: The output directory for the processed dataset.
- `--tokenizer`: The tokenizer to use. In this case, we will use the Llama 3.1 tokenizer.
- `--num_workers`: The number of workers to use for processing. This should be set to the number of CPU cores available.
- `--splits`: The splits to process, e.g. train, test, validation. The Dolmino Mix dataset has only the train split.
- `--concat_tokens`: The number of tokens to concatenate. This should be set to the maximum number of tokens that can be processed by the model.
- max_tokens: The maximum number of tokens to process.
- `--val_tokens`: The number of tokens to use for validation. This is useful when the dataset doesn't have a validation split. Tokens will be picked from the remaining tokens after the training split (if available).
- `--shuffle`: Whether to shuffle the dataset.

#### DCLM

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset dclm \
    --out_root $OUTPUT_FOLDER/dclm \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 23600000000 \
    --val_tokens 4720000 \
    --shuffle
```

#### Flan

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset flan \
    --out_root $OUTPUT_FOLDER/flan \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 8300000000 \
    --val_tokens 1660000 \
    --shuffle
```

#### PES2O

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 \
    --data_subset pes2o \
    --out_root $OUTPUT_FOLDER/pes2o \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 2925000000 \
    --val_tokens 585000 \
    --shuffle
```

#### Wiki

```bash
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset wiki \
    --out_root $OUTPUT_FOLDER/wiki \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 3555000000 \
    --val_tokens 711000 \
    --shuffle
```

#### Stackexchange

```bash
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset stackexchange \
    --out_root $OUTPUT_FOLDER/stackexchange \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 1225000000 \
    --val_tokens 245000 \
    --shuffle
```

#### Math

```bash
python scripts/data_prep/convert_dataset_hf.py \
    --dataset allenai/dolmino-mix-1124 \
    --data_subset math \
    --out_root $OUTPUT_FOLDER/math \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_workers 16 \
    --splits train \
    --concat_tokens 8192 \
    --max_tokens 10400000000 \
    --val_tokens 2080000 \
    --shuffle
```

### PDFs

Number of pdfs: 675661

## Training

In this section, we will describe how to train the model with LLM-Foundry.
We will use the `train.py` script to train the model, wrapping it with `compose` for distributed training.

```bash
compose train.py
```
