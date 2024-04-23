#!/usr/bin/env python3

"""
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
"""

import csv
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, Sampler, BatchSampler, RandomSampler, DataLoader
from typing import Dict, Iterator, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import random
import os
import torch
from minbert_tokenizer import tokenizer


def preprocess_string(s):
    return " ".join(
        s.lower()
        .replace(".", " .")
        .replace("?", " ?")
        .replace(",", " ,")
        .replace("'", " '")
        .split()
    )


def load_multitask_data(
    sentiment_filename, paraphrase_filename, similarity_filename, split="train"
):
    sentiment_data = []
    num_labels = {}
    if split == "test":
        with open(sentiment_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent = record["sentence"].lower().strip()
                sent_id = record["id"].lower().strip()
                label = int(record["sentiment"].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label, sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == "test":
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                paraphrase_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )

    else:
        with open(paraphrase_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                try:
                    sent_id = record["id"].lower().strip()
                    paraphrase_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            int(float(record["is_duplicate"])),
                            sent_id,
                        )
                    )
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == "test":
        with open(similarity_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        sent_id,
                    )
                )
    else:
        with open(similarity_filename, "r") as fp:
            for record in csv.DictReader(fp, delimiter="\t"):
                sent_id = record["id"].lower().strip()
                similarity_data.append(
                    (
                        preprocess_string(record["sentence1"]),
                        preprocess_string(record["sentence2"]),
                        float(record["similarity"]),
                        sent_id,
                    )
                )

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data


data_dir = "data"
# We use string to identity different task
# Currently the tasks are "sentiment_analysis", "paraphrase" and "similarity"
TaskId = str
# The index returned by the sampler, it contains a task id and indices for the subset.
DatasetIndex = Tuple[TaskId, List[int]]
all_task_ids = ["sentiment_analysis", "paraphrase", "similarity"]


all_dataset = {}
for split in ["train", "dev"]:
    sentiment_filename = os.path.join(data_dir, f"ids-sst-{split}.csv")
    paraphrase_filename = os.path.join(data_dir, f"quora-{split}.csv")
    similarity_filename = os.path.join(data_dir, f"sts-{split}.csv")
    sentiment_data, num_labels, paraphrase_data, similarity_data = load_multitask_data(
        sentiment_filename, paraphrase_filename, similarity_filename, split=split
    )
    dd = {}
    dd["sentiment_analysis"] = sentiment_data
    dd["paraphrase"] = paraphrase_data
    dd["similarity"] = similarity_data
    all_dataset[split] = dd

"""
Overview of Dataset, Dataloader and Sampler
`torch.utils.data` provides three useful interface for iterating dataset. 
You can get information about them in pytorch documentation.
1. `Dataset` is an index-able data structure like a dictionary. You can use any value as the key, not limited to int.
2. `DataLoader` is an iterable data structure (not indexable). It's used to assemble data from Dataset and should return batches of data
3. `Sampler` is used by `DataLoader` to sample data from the `Dataset`.
"""

from abc import ABC, abstractmethod


class SingleTaskDataset(ABC, Dataset):
    def __init__(self, task_id: TaskId, version: str):
        assert task_id in all_task_ids, f"{task_id} is not in {all_task_ids}"
        assert version in ["train", "dev"], f"{version} is not in {'train', 'dev'}"
        self.dataset = all_dataset[version][task_id]
        self.task_id = task_id
        self.version = version

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int):
        return self.dataset[i]

    @abstractmethod
    def collate_fn(self, data):
        raise NotImplementedError()


class SingleSentenceDataset(SingleTaskDataset):
    def __init__(self, task_id: str, version: str):
        assert task_id == "sentiment_analysis"
        super().__init__(task_id, version)

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(data)

        batched_data = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            #'sents': sents,
            "sent_ids": sent_ids,
        }

        return batched_data


"""
This is the implementation for paraphrase task
Two sentences are concatenated and tokenized
"""
class ConcatSentencePairDataset(SingleTaskDataset):
    def __init__(self, task_id: str, version: str):
        assert task_id == "paraphrase"
        super().__init__(task_id, version)

    def collate_fn(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]
        # Tokenize the concatenated sentences
        encoding = tokenizer(
            sent1, sent2, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]
        labels = torch.LongTensor(labels)
        # the name here should be the same as the forward function you use!!!
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sent_ids": sent_ids,
        }

"""
This is the implementation for similarity task
Two sentences are concatenated and tokenized now.
BUT THIS IS INCORRECT!!!
TODO : Replace this with the correct implementation
"""
class SeparateSentencePairDataset(SingleTaskDataset):
    def __init__(self, task_id: str, version: str):
        assert task_id == "similarity"
        super().__init__(task_id, version)

    def collate_fn(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]
        # Tokenize the concatenated sentences
        encoding = tokenizer(
            sent1, sent2, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]
        labels = torch.FloatTensor(labels)
        # the name here should be the same as the forward function you use!!!
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sent_ids": sent_ids,
        }


"""
We define a torch Dataset, which is just a group of list.
"""


class MultiTaskDataset(Dataset):
    def __init__(self, dataset_map: Dict[TaskId, SingleTaskDataset]):
        self.dataset_map = dataset_map
    def __len__(self):
        return sum([len(ds) for ds in self.dataset_map.values()])
    def __getitem__(self, i: DatasetIndex):
        task_id, indices = i
        ds = self.dataset_map[task_id]
        if isinstance(indices, int):
            return (task_id, ds[indices])
        else:
            return (task_id, [ds[i] for i in indices])

"""
A sampler that samples dataset of different tasks one by one (in the order given by the array `order`)
It returns indices which are used to index into MultiTaskDataset.
"""
class MultiTaskSequentialBatchSampler(Sampler):
    def __init__(
        self,
        dataset: MultiTaskDataset,
        order: List[TaskId],
        batch_size: int,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.order = order

    def __len__(self):
        total = 0
        for i in self.dataset.dataset_map:
            total += len(self.dataset.dataset_map[i]) // self.batch_size
        return total

    def __iter__(self) -> Iterator[DatasetIndex]:
        # we simply create a sampler for each task one by one
        # then sample with each sampler
        m = self.dataset.dataset_map
        for task_id in self.order:
            sampler = BatchSampler(
                RandomSampler(m[task_id]), self.batch_size, self.drop_last
            )
            for sample in sampler:
                yield (task_id, sample)


class MultiTaskRoundRobinSampler(Sampler):
    def __init__(
        self,
        dataset: MultiTaskDataset,
        order: List[TaskId],
        batch_size: int,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.order = order

    def __len__(self):
        total = 0
        for i in self.dataset.dataset_map:
            total += len(self.dataset.dataset_map[i]) // self.batch_size
        return total

    def __iter__(self) -> Iterator[DatasetIndex]:
        # we create mutiple samplers for each task
        # then sample with each sampler randomly
        # after exhausting one sampler, we remove it from the list
        m = self.dataset.dataset_map
        samplers = [
            (
                task_id,
                iter(
                    BatchSampler(
                        RandomSampler(m[task_id]), self.batch_size, self.drop_last
                    )
                ),
            )
            for task_id in self.order
        ]
        while len(samplers) > 0:
            index = random.randrange(0, len(samplers))
            task_id, sampler = samplers[index]
            try:
                sample = next(sampler)
                yield (task_id, sample)
            except StopIteration:
                samplers.pop(index)
        return None


"""
Prepare the dataset.
Let's use the default one first we will use the glue dataset eventually
`all_dataset` stores both the training data and validation data
"""
# create a training dataset
# evaluation dataset is handled differently
training_dataset_map: Dict[TaskId, SingleTaskDataset] = {
    "sentiment_analysis": SingleSentenceDataset("sentiment_analysis", "train"),
    "paraphrase": ConcatSentencePairDataset("paraphrase", "train"),
    "similarity": SeparateSentencePairDataset("similarity", "train"),
}

eval_dataset_map: Dict[TaskId, SingleTaskDataset] = {
    "sentiment_analysis": SingleSentenceDataset("sentiment_analysis", "dev"),
    "paraphrase": ConcatSentencePairDataset("paraphrase", "dev"),
    "similarity": SeparateSentencePairDataset("similarity", "dev"),
}

training_dataset = MultiTaskDataset(training_dataset_map)
# there is no multi-task evaluation dataset, because the dataset is evaluated separately
