import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument(
        "--output-folder", type=str, help="Output folder", required=True
    )
    arg_parser.add_argument("--resume", action="store_true", help="Resume download")
    arg_parser.add_argument("--allow-patterns", type=str, help="Allow patterns")
    arg_parser.add_argument("--max-workers", type=int, help="Max workers", default=4)
    args = arg_parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    folder = snapshot_download(
        args.dataset,
        repo_type="dataset",
        local_dir=str(output_folder),
        allow_patterns=args.allow_patterns,
        resume_download=args.resume,
        max_workers=args.max_workers,
    )

    print(f"Downloaded dataset to {folder}")
