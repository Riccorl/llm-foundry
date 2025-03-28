## Data Processing

### Dolmino

50B tokens mix as per Olmo 2 paper:

```python
portions = {
    "dclm": 0.472, 
    "flan": 0.166,
    "pes2o": 0.0585,
    "wiki": 0.0711,
    "stackexchange": 0.0245,
    "math": 0.208,
}

print("Training tokens:")
TARGET_TOKENS = 50_000_000_000
for k, v in portions.items():
    print(f"{k}: {int(v * TARGET_TOKENS)}")

print()

print("Validation tokens:")
TARGET_VAL_TOKENS = 10_000_000
for k, v in portions.items():
    print(f"{k}: {int(v * TARGET_VAL_TOKENS)}")
```

### DCLM

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset dclm --out_root /home/riccar_orlando/data/dolmino/processed/dclm  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 23600000000 --shuffle 
```

### Flan

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset flan --out_root /home/riccar_orlando/data/dolmino/processed/flan  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 8300000000 --shuffle
```

### PES2O

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset pes2o --out_root /home/riccar_orlando/data/dolmino/processed/pes2o  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 2925000000 --shuffle
```

### Wiki

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset wiki --out_root /home/riccar_orlando/data/dolmino/processed/wiki  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 3555000000 --shuffle
```

### Stackexchange

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset stackexchange --out_root /home/riccar_orlando/data/dolmino/processed/stackexchange  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 1225000000 --shuffle
```

### Math

```bash
python scripts/data_prep/convert_dataset_hf.py --dataset allenai/dolmino-mix-1124 --data_subset math --out_root /home/riccar_orlando/data/dolmino/processed/math  --tokenizer meta-llama/Llama-3.1-8B-Instruct --num_workers 16 --splits train --concat_tokens 8192 --max_tokens 10400000000 --shuffle
```
