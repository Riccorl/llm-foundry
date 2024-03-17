from pathlib import Path
from datasets import load_dataset
import os

import psutil

if __name__ == "__main__":
    # set hf cache to scratch
    # os.environ["HF_DATASETS_CACHE"] = str(Path(os.getenv("SCRATCH")) / "hf_cache")
    # path = str(Path(os.getenv("RED_PAJAMA_HOME")) / "it" /"documents" / "2023-14")
    # pattern = "*" # "*_head.json" / "*_middle.json"
    # data_files = glob(f'{path}/**/{pattern}.json')
    # # order data files by name
    # data_files.sort()
    # hf_dataset = load_dataset("json", data_files=data_files, cache_dir=False, num_proc=psutil.cpu_count(logical=True))

    data = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        # path,
        name="default",
        partition="head_middle",
        snapshots=["2023-14"],
        languages=["it"],
        cache_dir=str(Path(os.getenv("SCRATCH")) / "hf_cache"),
        num_proc=psutil.cpu_count(logical=True),
        streaming=True,
    )
