import argparse
from glob import glob
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Union

import huggingface_hub
import psutil
import datasets as hf_datasets
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def main(data_folder, file_to_save, max_samples):
#     progress_bar = tqdm(total=max_samples)
#     i = 0
#     file_to_save = Path(file_to_save)
#     file_to_save.parent.mkdir(parents=True, exist_ok=True)
#     with open(file_to_save, "w") as f_out:
#         # ordered by name
#         for jsonl_path in sorted(Path(data_folder).glob("*.jsonl")):
#             print(f"Reading {jsonl_path}")
#             with open(jsonl_path, "r") as f_in:
#                 for line in f_in:
#                     if i >= max_samples:
#                         sys.exit(0)
#                     line = json.loads(line)
#                     f_out.write(f"{line['text']}\n")
#                     i += 1
#                     progress_bar.update(1)

def build_hf_dataset(
    dataset_name: str,
    split: str,
    data_type: str = "json",
    data_subset: Union[str, None] = None,
    streaming: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: Optional[int] = None,
) -> Dataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating, whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS, the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns:
        An IterableDataset.
    """
    is_local = os.path.exists(dataset_name)
    if is_local:
        if os.path.isdir(dataset_name):
            # infer data type from file extension
            # data_type = dataset_name.split(".")[-1]
            data_files = glob(f'{dataset_name}/*.{data_type}')
        else:
            data_files = dataset_name
        hf_dataset = hf_datasets.load_dataset(
            data_type,
            data_files=data_files,
            split=split,
            streaming=streaming,
            num_proc=num_workers if not streaming else None,
        )
    else:
        hf_dataset = hf_datasets.load_dataset(
            path=dataset_name, name=data_subset, split=split, streaming=streaming
        )
    if shuffle:
        print("Shuffling dataset")
        hf_dataset = hf_dataset.shuffle(seed=seed)
    return hf_dataset

def main(dataset, output_file, max_samples, data_type, streaming=False):
    hf_dataset = build_hf_dataset(dataset, "train", data_type, num_workers=4, streaming=streaming)
    progress_bar = tqdm(total=max_samples)
    i = 0
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f_out:
        # ordered by name
        for line in hf_dataset:
            if i >= max_samples:
                sys.exit(0)
            f_out.write(f"{line['text']}\n")
            i += 1
            progress_bar.update(1)
        # for jsonl_path in sorted(Path(data_folder).glob("*.jsonl")):
            # print(f"Reading {jsonl_path}")
            # with open(jsonl_path, "r") as f_in:
            #     for line in f_in:
            #         if i >= max_samples:
            #             sys.exit(0)
            #         line = json.loads(line)
            #         f_out.write(f"{line['text']}\n")
            #         i += 1
            #         progress_bar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        # default="uonlp/CulturaX",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        # default="./",
        help="Path to save the dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Max number of row to download",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Download in streaming mode",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="json",
        help="Data type of the dataset",
    )
    args = parser.parse_args()

    main(**vars(args))
