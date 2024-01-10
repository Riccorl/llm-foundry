import argparse
import json
import logging
import os
from pathlib import Path
import sys

import huggingface_hub
import psutil
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(data_folder, file_to_save, max_samples):
    progress_bar = tqdm(total=max_samples)
    i = 0
    file_to_save = Path(file_to_save)
    file_to_save.parent.mkdir(parents=True, exist_ok=True)
    with open(file_to_save, "w") as f_out:
        # ordered by name
        for jsonl_path in sorted(Path(data_folder).glob("*.jsonl")):
            print(f"Reading {jsonl_path}")
            with open(jsonl_path, "r") as f_in:
                for line in f_in:
                    if i >= max_samples:
                        sys.exit(0)
                    line = json.loads(line)
                    f_out.write(f"{line['text']}\n")
                    i += 1
                    progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF Dataset")
    parser.add_argument(
        "--data_folder",
        type=str,
        # default="uonlp/CulturaX",
        help="Dataset to download.",
    )
    # parser.add_argument(
    #     "--language",
    #     type=str,
    #     default="it",
    #     help="Language of the dataset to download.",
    # )
    parser.add_argument(
        "--file_to_save",
        type=str,
        # default="./",
        help="Path to save the dataset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max number of row to download",
    )
    # parser.add_argument(
    #     "--step",
    #     type=int,
    #     default=10,
    #     help="Number of document to download to wait every memory check",
    # )
    # parser.add_argument(
    #     "--streaming",
    #     action="store_true",
    #     help="Download in streaming mode",
    # )
    # parser.add_argument(
    #     "--split_size",
    #     type=int,
    #     default=100_000,
    #     help="Number of document per file during jsonl saving",
    # )
    args = parser.parse_args()

    main(**vars(args))
