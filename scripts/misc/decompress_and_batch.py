import os
import json
import gzip
from pathlib import Path
import tqdm
import concurrent.futures
import psutil


def process_file_chunk(file_path, output_folder):
    # f_out = open(os.path.join(output_folder, f"batch_{batch_index}.jsonl"), "w")
    # for f in files_chunk:
    # base_name = os.path.basename(file_path)
    file_path = Path(file_path)
    output_folder = Path(output_folder)
    # get the folders until there is a match with the output folder
    base_folder = file_path.parent
    while base_folder.name != output_folder.name and base_folder != Path("/"):
        base_folder = base_folder.parent
    # now get the base name without the extension
    base_name = file_path.stem
    # and all the folders after the base folder
    base_folder = file_path.parent.relative_to(base_folder)
    # add the base folder name to the output folder
    output_folder = Path(output_folder) / base_folder
    # create the output folder if it does not exist
    output_folder.mkdir(exist_ok=True, parents=True)
    # create the output path
    output_path = output_folder / base_name
    # check if the output file already exists, if so, skip
    if output_path.exists():
        return
    with gzip.open(file_path, "rt") as gz, open(output_path, "w") as f_out:
        # decompress and write to output file
        for line in gz:
            f_out.write(line)


def process_files_in_batches(input_folder, output_folder, max_workers=4):

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in tqdm.tqdm(os.walk(input_folder)):
            for f in files:
                # files_chunk.append(os.path.join(root, f))
                # if len(files_chunk) == batch_size:
                full_path = os.path.join(root, f)
                futures.append(executor.submit(process_file_chunk, full_path, output_folder))

                # Ensure we don't have too many futures at once
                if len(futures) >= max_workers:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()  # To catch any exceptions
                    futures = []

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()  # To catch any exceptions


if __name__ == "__main__":
    print(f"Running {psutil.cpu_count(logical=True)} workers")
    process_files_in_batches(
        # number of files
        input_folder="/leonardo_scratch/large/userexternal/phuguetc/redpajamav2/it/documents_deduped_by_CC_fineweb",
        # number of files 839996
        output_folder="/leonardo_scratch/large/userexternal/rorland1/data/redpajamav2/it/documents_deduped_by_CC_fineweb",
        # batch_size=1000000,
        # max_workers=psutil.cpu_count(logical=True),
        max_workers=16,
    )
    print("Done")
