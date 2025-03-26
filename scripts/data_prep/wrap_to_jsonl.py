import argparse
import json
from pathlib import Path
from tqdm import tqdm


def wrap_to_jsonl(input_folder: str, output_folder: str) -> None:
    """Wrap files in input_folder into jsonl format and save them to output_folder.

    Args:
        input_folder (str): Path to the input folder containing text files.
        output_folder (str): Path to the output folder where jsonl files will be saved.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    for file in tqdm(input_path.iterdir(), desc="Processing files"):
        if file.is_file() and file.suffix != ".json":
            with open(file, "r") as f:
                content = f.read()
                jsonl_content = {"text": content}
                output_file = output_path / f"{file.stem}.jsonl"
                with open(output_file, "w") as out_f:
                    out_f.write(json.dumps(jsonl_content) + "\n")


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_folder", type=str, help="Output folder")
    args = arg_parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    wrap_to_jsonl(args.input_folder, args.output_folder)
