import datasets
from datasets import Dataset
from transformers import AutoTokenizer

# num_sequence_wanted = 20000
max_seq_len = 8192
portions = {
    "dclm": 0.472 * 50000000000,
    "flan": 0.166 * 50000000000,
    "pes2o": 0.0585 * 50000000000,
    "wiki": 0.0711 * 50000000000,
    "stackexchange": 0.0245 * 50000000000,
    "math": 0.208 * 50000000000,
}
seed = 42
save_path = "/home/riccar_orlando/data/dolmino-mix-1124-50"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

dolmino = datasets.load_dataset("allenai/dolmino-mix-1124", streaming=True)
sources = ["dclm", "flan", "math", "pes2o", "stackexchange", "wiki"]

source_datasets = dict()
for source in sources:
    source_datasets[source] = datasets.load_dataset(
        "allenai/dolmino-mix-1124",
        streaming=True,
        data_dir="data/" + source,
        split="train",
    )

shuffled_datasets = dict()
for source in sources:
    shuffled_datasets[source] = source_datasets[source].shuffle(seed=seed)

iterators = dict()
for source in sources:
    iterators[source] = iter(shuffled_datasets[source])

result_dataset = {"id": [], "text": [], "added": [], "created": []}

total_len_added = {
    "dclm": 0,
    "flan": 0,
    "math": 0,
    "pes2o": 0,
    "stackexchange": 0,
    "wiki": 0,
    "total": 0,
}


def add_sample(sample, source):
    result_dataset["id"].append(sample["id"])
    result_dataset["text"].append(sample["text"])
    result_dataset["added"].append(sample["added"])
    result_dataset["created"].append(sample["created"])
    sample_len = len(
        tokenizer(sample["text"], truncation=True, max_length=max_seq_len)["input_ids"]
    )
    total_len_added["total"] += sample_len
    total_len_added[source] += sample_len


for source in sources:
    while total_len_added[source] < portions[source]:
        try:
            sample = next(iterators[source])
        except StopIteration:
            break
        add_sample(sample, source)

while total_len_added["total"] < 50000000000:
    try:
        sample = next(iterators["dclm"])
    except StopIteration:
        break
    add_sample(sample, "dclm")
print("Total length added:", total_len_added)

result_dataset_hf = Dataset.from_dict(result_dataset)

result_dataset_hf = result_dataset_hf.shuffle(seed=seed)
result_dataset_hf.save_to_disk(save_path)
