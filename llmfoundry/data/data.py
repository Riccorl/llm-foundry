# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, List, Union

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

__all__ = [
    "ConcatTokensDataset",
    "NoConcatDataset",
]


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
    ):
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # convert to bytes to store in MDS binary format
            yield {"text": sample["text"].encode("utf-8")}


class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        write_batch_size: int = 10_000,
        multi_process: bool = False,
        filter_by_domain: List | None = None,
        filter_by_length: int | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false" if not multi_process else "true"
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.write_batch_size = write_batch_size
        self.multi_process = multi_process

        if filter_by_domain is None:
            filter_by_domain = []
        else:
            print(f"Filtering by domain: {filter_by_domain}")
        self.filter_by_domain = filter_by_domain
        self.filter_by_length = filter_by_length

        self.bos_tokens = self.tokenizer(
            self.bos_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        if len(self.bos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --bos_text, but your BOS text is not tokenizing to one token\
                , instead we got {self.bos_tokens}. Quit if this was in error.",
            )

        self.eos_tokens = self.tokenizer(
            self.eos_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f"You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.",
            )

        eos_text_provided = self.eos_text != ""
        bos_text_provided = self.bos_text != ""
        test_text = self.tokenizer("")
        if len(
            test_text["input_ids"],
        ) > 0 and (eos_text_provided or bos_text_provided):
            message = (
                "both eos and bos"
                if eos_text_provided and bos_text_provided
                else ("eos_text" if eos_text_provided else "bos_text")
            )
            warnings.warn(
                f"The provided tokenizer adds special tokens, but you also specified {message}. This may result "
                + "in duplicated special tokens. Please be sure this is what you intend.",
            )

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        if self.multi_process:
            return self.multi_threaded_iter()
        else:
            return self.single_threaded_iter()

    def single_threaded_iter(self) -> Iterable[Dict[str, bytes]]:
        total_samples = 0
        buffer = []
        for sample in self.hf_dataset:
            if self.filter_by_domain:
                # culturax
                if "url" in sample.keys():
                    if any(
                        excluded_domain in sample["url"]
                        for excluded_domain in self.filter_by_domain
                    ):
                        continue
                    # redpajama
                elif "metadata" in sample.keys() and "url" in sample["metadata"].keys():
                    if any(
                        excluded_domain in sample["metadata"]["url"]
                        for excluded_domain in self.filter_by_domain
                    ):
                        continue
                else:
                    print(
                        "No domain information found in dataset, available keys: ",
                        sample.keys(),
                    )

            text = None
            if "text" in sample:
                text = sample["text"]
            elif "raw_content" in sample:
                text = sample["raw_content"]
            else:
                raise ValueError(
                    "No text field found in dataset, available keys: ", sample.keys()
                )
            encoded = self.tokenizer(text, truncation=False, padding=False)
            if (
                self.filter_by_length
                and len(encoded["input_ids"]) <= self.filter_by_length
            ):
                continue
            total_samples += 1
            iids = encoded["input_ids"]
            buffer = buffer + self.bos_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[: self.max_length]
                buffer = buffer[self.max_length :] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes(),
                    "num_tokens": len(concat_sample),
                }
        print(f"Total samples processed: {total_samples}")

    def multi_threaded_iter(self) -> Iterable[Dict[str, bytes]]:
        total_samples = 0

        print("Starting multi-threaded tokenization")

        buffer = []
        # split samples into smaller chunks to do parallel tokenization
        shards = self.hf_dataset.num_rows // self.write_batch_size + 1
        print("Shards: ", shards)
        for i in range(shards):
            shard = self.hf_dataset[
                i * self.write_batch_size : (i + 1) * self.write_batch_size
            ]

            if "text" in shard:
                text_in_shard = shard["text"]
            elif "raw_content" in shard:
                text_in_shard = shard["raw_content"]
            else:
                raise ValueError(
                    "No text field found in dataset, available keys: ", shard.keys()
                )
            # text_in_shard = shard["text"]

            # filter by domain
            if self.filter_by_domain:
                _text_in_shard = []
                for i, text in enumerate(text_in_shard):
                    # culturax
                    if "url" in shard:
                        if not any(
                            excluded_domain in shard["url"][i]
                            for excluded_domain in self.filter_by_domain
                        ):
                            _text_in_shard.append(text)
                    # redpajama
                    elif "metadata" in shard and "url" in shard["metadata"]:
                        if not any(
                            excluded_domain in shard["metadata"]["url"][i]
                            for excluded_domain in self.filter_by_domain
                        ):
                            _text_in_shard.append(text)
                    else:
                        print(
                            "No domain information found in dataset, available keys: ",
                            shard.keys(),
                        )
                        _text_in_shard.append(text)
                text_in_shard = _text_in_shard

            encoded_shard = self.tokenizer(
                text_in_shard,
                truncation=False,
                padding=False,
            )
            # filter by length
            if self.filter_by_length:
                encoded_shard = [
                    encoded
                    for encoded in encoded_shard
                    if len(encoded["input_ids"]) >= self.filter_by_length
                ]

            total_samples += len(encoded_shard["input_ids"])
            for encoded in encoded_shard["input_ids"]:
                iids = encoded  # ['input_ids']
                buffer = buffer + self.bos_tokens + iids + self.eos_tokens
                while len(buffer) >= self.max_length:
                    concat_sample = buffer[: self.max_length]
                    buffer = buffer[self.max_length :] if self.should_wrap else []
                    yield {
                        # convert to bytes to store in MDS binary format
                        "tokens": np.asarray(concat_sample).tobytes(),
                        "num_tokens": len(concat_sample),
                    }

        print(f"Total samples processed: {total_samples}")
