import argparse
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_folder", type=str)
    arg_parser.add_argument("output_folder", type=str, help="Output folder")
    arg_parser.add_argument(
        "--languages",
        type=str,
        default="en",
        help="Comma-separated list of languages",
    )
    arg_parser.add_argument("--threshold", type=float, default=0.95)
    arg_parser.add_argument("--tasks", type=int, default=1)
    arg_parser.add_argument("--workers", type=int, default=-1)
    arg_parser.add_argument("--logging_dir", type=str, default="logs/")
    args = arg_parser.parse_args()

    languages = args.languages.split(",") if args.languages else None
    if languages is None:
        raise ValueError("No languages provided")

    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output folder: {args.output_folder}")
    logger.info(f"Languages: {languages}")
    logger.info(f"Threshold: {args.threshold}")

    pipeline = [
        JsonlReader(data_folder=args.input_folder, file_progress=True),
        LanguageFilter(languages=languages, language_threshold=args.threshold),
        JsonlWriter(output_folder=args.output_folder, compression=None),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=args.logging_dir,
        tasks=args.tasks,
        workers=args.workers,
    )
    executor.run()
