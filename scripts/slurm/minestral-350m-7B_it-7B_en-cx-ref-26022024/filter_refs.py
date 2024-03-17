from glob import glob
import json
import os
import argparse
import re

from tqdm import tqdm


def main(path, data_files_pattern, out_root):
    if os.path.isdir(path):
        match_string = f"{path}/{data_files_pattern}"
        data_files = [f for f in glob(f"{path}/*") if re.search(match_string, f)]
        # order data files by name
        data_files.sort()
    else:
        data_files = path
    
    for data_file in tqdm(data_files, desc="Filtering data"):
        with open(data_file, "r") as f:
            data = [json.loads(line) for line in f]
        
        filtered_data = []
        for sample in data:
            if sample["predictions"]["__label__wiki"] > sample["predictions"]["__label__cc"]:
                filtered_data.append(sample)

        out_file = os.path.join(out_root, os.path.basename(data_file))
        with open(out_file, "w") as f:
            for sample in filtered_data:
                f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--data_files_pattern", type=str, default="*")
    args = parser.parse_args()

    main(**vars(args))
