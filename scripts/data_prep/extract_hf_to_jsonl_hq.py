import argparse
import json
import logging
import os
from pathlib import Path
import psutil
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from random import randrange
from pathlib import Path
import random
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_CACHE = os.getenv("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface"))

HIGH_QUALITY_SOURCES = ["wikipedia.org", "reddit.com", "eur-lex.europa.eu", "gutenberg.org", "wikinews.org"]


def save_file(dataset_path, hq_subset, shard_idx_hq):
    split_path_hq = dataset_path / f"hq"
    split_path_hq.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {split_path_hq / f'{shard_idx_hq}.jsonl'}")
    print(f"Writing {len(hq_subset)} lines in hq based ")
    with open(split_path_hq / f"{shard_idx_hq}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in hq_subset)


def save_to_jsonl(dataset, path_to_save, split_size, sources, max_iterations):
    # now we can save the dataset in jsonl format, divided in multiple files
    # we would like to parallelize this step, we can use the multiprocessing library
    dataset_path = Path(path_to_save)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # create a new file for each split, parallelize this step
    hq_subset = []
    shard_idx_hq = 0

    for i, sample in tqdm(enumerate(dataset)):

        if i == max_iterations: # stop after max_iterations reached
            break

        # sample only certain resources
        if sources is not None and sample["source"] not in sources:
           continue
       
        base_url = urlparse(sample["url"]).netloc

        if any([q in base_url for q in HIGH_QUALITY_SOURCES]):
            hq_subset.append(sample)
        else:
            continue
        
        if len(hq_subset) >= split_size:
            save_file(dataset_path, hq_subset, shard_idx_hq)

            hq_subset = []

            shard_idx_hq += 1

    if len(hq_subset) > 0:
        save_file(dataset_path, hq_subset, shard_idx_hq)


def main(args):
    logger.info("==== Starting download Dataset ====")
   
    if args.max_iterations == -1:
        # to get the real number of iterations, used to create the reservoir index, i need to query the dataset.
        max_iterations = load_from_disk(
            args.dataset_path,
            #     split=args.ds_split,
            #     cache_dir=HF_CACHE,
            #     token=True,
            #     num_proc=psutil.cpu_count(logical=False),
            ).num_rows[args.ds_split]
    else:
        max_iterations = int(args.max_iterations)
   

    dataset = load_dataset(
        args.dataset_path,
        split=args.ds_split,
        cache_dir=HF_CACHE,
        streaming=True
    )

    save_to_jsonl(dataset, args.path_to_save, args.split_size, args.sources, max_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HF Dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Dataset to download.",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="./",
        help="Path to save the dataset.",
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
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=-1,
        help="number of iterations to do at most"
    )
    args = parser.parse_args()

    main(args)
