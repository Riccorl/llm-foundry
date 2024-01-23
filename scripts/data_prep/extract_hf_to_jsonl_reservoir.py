import argparse
import json
import logging
import os
from pathlib import Path
import psutil
from datasets import load_dataset
from tqdm import tqdm
from random import randrange
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HF_CACHE = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
URLS_FILE_NAME = "urls.json"


def reservoir_streaming(dataset, max_samples, sources, url_file_path):
    reservoir_indexes = None

    reservoir_indexes = []

    with open(url_file_path, "w") as url_file:

        for i, sample in tqdm(enumerate(dataset)):

            # save the url 
            url_file.write(json.dumps({"index": i, "url": sample["url"]}) + "\n")

            # sample only certain resources
            if sources is not None and sample["source"] not in sources:
                continue
            
            # --- init the reservoir
            if len(reservoir_indexes) < max_samples:
                reservoir_indexes.append(i)
                continue
            # ---
            
            # --- reservoir sample
            j = randrange(i)
            if j < max_samples:
                reservoir_indexes[j] = i
            # ---

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
    #token = huggingface_hub.HfFolder.get_token()
    #if token is None:
    #    logger.info("HuggingFace Login")
    #    login()

    logger.info("==== Starting download Dataset ====")
    dataset = load_dataset(
        path=args.dataset_path,
        split=args.ds_split,
        cache_dir=HF_CACHE,
    #    token=True,
        num_proc=psutil.cpu_count(logical=False),
    )

    # sources = ["mC4", "OSCAR-2301", "OSCAR-2201"]

    # create path for the URLs file

    if not os.path.exists(args.url_file_path):
        os.makedirs(args.url_file_path)

    url_file_path = Path(args.url_file_path) / URLS_FILE_NAME

    reservoir_indexes  = reservoir_streaming(dataset, args.max_samples, args.sources, url_file_path)

    save_to_jsonl(dataset, args.path_to_save, args.split_size, args.max_samples, reservoir_indexes, args.ds_split)


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
    parser.add_argument(
        "--url_file_path",
        type=str,
        default="",
        help="path to save the url file"
    )
    args = parser.parse_args()

    main(args)
