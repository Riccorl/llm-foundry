from datasets import load_dataset
from urllib.parse import urlsplit
import psutil
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import json


HIGH_QUALITY_SOURCES = ["wikipedia.org", "reddit.com", "eur-lex.europa.eu", "statm.org", "gutenberg.org", "wikinews.org"]

def save_to_jsonl(dataset, path_to_save, split_size, sources):
    # now we can save the dataset in jsonl format, divided in multiple files
    # we would like to parallelize this step, we can use the multiprocessing library
    dataset_path = Path(path_to_save)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # create a new file for each split, parallelize this step
    web_subset = []
    shard_idx_web = 0

    for i, sample in tqdm(enumerate(dataset)):
        
        base_url = urlparse(sample["url"]).netloc
        
        if sample["source"] not in sources:
            continue 

        if "wikinews.org" in base_url:
            print("Wikinews url -> ", base_url)

        if not any([s in base_url for s in HIGH_QUALITY_SOURCES]):
            web_subset.append(sample)
        else:
            continue

        if len(web_subset) >= split_size:
            split_path_web = dataset_path / f"web_cleaned"
            split_path_web.mkdir(parents=True, exist_ok=True)

            print(f"Writing to {split_path_web / f'{shard_idx_web}.jsonl'}")
            print(f"Writing {len(web_subset)} lines in web based ")
            with open(split_path_web / f"{shard_idx_web}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in web_subset)
            
            web_subset = []
            shard_idx_web += 1

    if len(web_subset) > 0:
        split_path_web = dataset_path / f"web_cleaned"
        split_path_web.mkdir(parents=True, exist_ok=True)

        print(f"Writing to {split_path_web / f'{shard_idx_web}.jsonl'}")
        print(f"Writing {len(web_subset)} lines in web based ")
        with open(split_path_web / f"{shard_idx_web}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in web_subset)


if __name__ == "__main__":
   
    dataset = load_dataset("/leonardo_work/IscrB_medit/culturax_res/reservoir_sample_10M_100M/it/web", num_proc=psutil.cpu_count(logical=False))

    path_to_save = "/leonardo_work/IscrB_medit/culturax_res/reservoir_sample_10M_100M/it/"

    sources = ["mC4", "OSCAR-2301", "OSCAR-2201"]

    save_to_jsonl(dataset["train"], path_to_save, 500_000, sources)
