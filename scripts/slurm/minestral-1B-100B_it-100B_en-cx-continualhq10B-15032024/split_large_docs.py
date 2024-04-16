import json
import sys

from tqdm import tqdm


if __name__ == "__main__":
    part = sys.argv[1]
    print(f"Splitting part {part}")
    with open(f"/leonardo_scratch/large/userexternal/phuguetc/books/book_part{part}") as f, open(
        f"/leonardo_scratch/large/userexternal/rorland1/books/book_split_{part}.jsonl", "w"
    ) as fo:
        n = 300
        for l in tqdm(f):
            sample = json.loads(l)
            if len(sample["text"].split(" ")) > n:
                _texts = sample["text"].split(" ")
                texts = [" ".join(_texts[i : i + n]) for i in range(0, len(_texts), n)]
            else:
                texts = [sample["text"]]
            for t in texts:
                sample["text"] = t
                fo.write(json.dumps(sample) + "\n")
