import argparse
import json
import logging
import os
from pathlib import Path

import huggingface_hub
import psutil
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm
from random import randrange

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(HF_CACHE)


def reservoir_streaming(dataset, max_samples, sources, ds_split="train"):
    reservoir_indexes = None

    reservoir_indexes = []

    if sources is None: # if the sources is None, we can loop over a fake range
        for i in tqdm(range(dataset[ds_split].dataset_size)):
            
            # init the reservoir
            if len(reservoir_indexes) < max_samples:
                reservoir_indexes.append(i)
                continue
            
            # reservoir sample
            j = randrange(i)
            reservoir_indexes[j] = i
    else:
        for i, sample in tqdm(enumerate(dataset[ds_split])):
            # sample only certain resources
            if sources is not None and sample["source"] not in sources:
                continue
            
            # init the reservoir
            if len(reservoir_indexes) < max_samples:
                reservoir_indexes.append(i)
                continue
            
            # reservoir sample
            j = randrange(i)
            reservoir_indexes[j] = i

    # return a uniform sample of indexes
    return reservoir_indexes


def save_to_jsonl(dataset, path_to_save, split_size, max_samples, reservoir_indexes, ds_split="train"):
    # now we can save the dataset in jsonl format, divided in multiple files
    # we would like to parallelize this step, we can use the multiprocessing library
    dataset_path = Path(path_to_save)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # create a new file for each split, parallelize this step
    dataset_subset = []
    saved = 0
    shard_idx = 0
    for index in reservoir_indexes:
        
        if saved >= max_samples:
            break

        sample = dataset[index]
        
        dataset_subset.append(sample)
        saved += 1

        if len(dataset_subset) >= split_size:

            split_path = dataset_path / f"{ds_split}"
            split_path.mkdir(parents=True, exist_ok=True)

            print(f"Writing to {split_path / f'{shard_idx}.jsonl'}")
            with open(split_path / f"{shard_idx}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in dataset_subset)

            dataset_subset = []
            shard_idx += 1


def main(args):
    token = huggingface_hub.HfFolder.get_token()
    if token is None:
        logger.info("HuggingFace Login")
        login()

    logger.info("==== Starting download CulturaX ====")
    dataset = load_dataset(
        path=args.dataset_path,
        # split="train",
        cache_dir=HF_CACHE,
        token=True,
        num_proc=psutil.cpu_count(logical=False),
    )

    # sources = ["mC4", "OSCAR-2301", "OSCAR-2201"]

    reservoir_indexes  = reservoir_streaming(dataset, args.max_samples, args.sources, ds_split="train")

    save_to_jsonl(dataset, args.path_to_save, args.split_size, args.max_samples, reservoir_indexes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF Dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        # default="uonlp/CulturaX",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="./",
        help="Path to save the dataset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max number of row to download",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=100_000,
        help="Number of document per file during jsonl saving",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs='*',
        default=None,
        help="Specific source to sample out from the datasets"
    )
    parser.add_argument(
        "--ds_split",
        type=str,
        default="train",
        help="dataset split to retrieve"
    )
    args = parser.parse_args()

    main(args)
