# from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=1000)
# for document in data_reader():
#     # do something with document
#     print(document)

###############################
# OR for a processing pipeline:
###############################

import os
from pathlib import Path
from huggingface_hub import snapshot_download

# from datatrove.executor import LocalPipelineExecutor
# from datatrove.pipeline.readers import ParquetReader
# from datatrove.pipeline.filters import LambdaFilter
# from datatrove.pipeline.writers import JsonlWriter

scratch_folder = os.getenv("SCRATCH", None)

if scratch_folder is None:
    raise ValueError("SCRATCH environment variable is not set")


scratch_folder = Path(scratch_folder)
download_folder = scratch_folder / "data" / "fineweb"
download_folder.mkdir(parents=True, exist_ok=True)


folder = snapshot_download(
    "HuggingFaceFW/fineweb",
    repo_type="dataset",
    local_dir=str(download_folder),
    # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
    allow_patterns="data/CC-MAIN-2022*",
)

# pipeline_exec = LocalPipelineExecutor(
#     pipeline=[
#         # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
#         ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10"),
#         LambdaFilter(lambda doc: "hugging" in doc.text),
#         JsonlWriter(str(download_folder), compression=None)
#     ],
#     tasks=1,
#     workers=4,
# )
# pipeline_exec.run()
