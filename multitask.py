import torch.utils
from transformers import (
    BertModel,
    BertTokenizer,
    BertPreTrainedModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler, RandomSampler, DataLoader
from typing import Dict, Iterator, List, Tuple, Optional, Any

from tqdm.autonotebook import tqdm

"""
AutoTokenizer will use a fast tokenizer (implemented in Rust) by default,
while BertTokenizer is slow.
"""
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

"""
Prepare the dataset.
Let's use the default one first we will use the glue dataset eventually
`all_dataset` stores both the training data and validation data
"""
import os
from minbert_datasets import load_multitask_data
data_dir = "data"
all_dataset = {}
for split in ["train", "dev"]:
    sentiment_filename = os.path.join(data_dir, f"ids-sst-{split}.csv")
    paraphrase_filename = os.path.join(data_dir, f"quora-{split}.csv")
    similarity_filename = os.path.join(data_dir, f"sts-{split}.csv") 
    sentiment_data, num_labels, paraphrase_data, similarity_data = load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split=split)
    dd = {}
    dd["sentiment_analysis"] = sentiment_data
    dd["paraphrase"] = paraphrase_data
    dd["similarity"] = similarity_data
    all_dataset[split] = dd


# We use string to identity different task
# Currently the tasks are "sentiment_analysis", "paraphrase" and "similarity"
TaskId = str
# The index returned by the sampler, it contains a task id and indices for the subset.
DatasetIndex = Tuple[TaskId, List[int]]


"""
Overview of Dataset, Dataloader and Sampler
`torch.utils.data` provides three useful interface for iterating dataset. 
You can get information about them in pytorch documentation.
1. `Dataset` is an index-able data structure like a dictionary. You can use any value as the key, not limited to int.
2. `DataLoader` is an iterable data structure (not indexable). It's used to assemble data from Dataset and should return batches of data
3. `Sampler` is used by `DataLoader` to sample data from the `Dataset`.
"""

"""
We define a torch Dataset, which is just a group of list.
"""
class MultiTaskDataset(Dataset):
    def __init__(self, dataset_map : Dict[TaskId, List]):
        self.dataset_map = dataset_map
    def __getitem__(self, i : DatasetIndex):
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
    def __init__(self, dataset : MultiTaskDataset, order : List[TaskId], batch_size : int, drop_last : bool):
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
            sampler = BatchSampler(RandomSampler(m[task_id]), self.batch_size, self.drop_last)
            for sample in sampler:
                yield (task_id, sample)   


"""
A brief explanation of what happens here
1. For loop will call `__iter__` method of the data loader
2. The data loader invokes the sampler's `__iter__` method (in this case, MultiTaskSequentialBatchSampler), to get a batch of indices
3. The indices are passed into the dataset's `__getitem__` method (in this case, MultiTaskDataset), to get a batch of data
4. The data are passed into the `collate_fn` function, which is resposible for packing everything into tensor.
"""

def single_sentence_pad__data(data):
    sents = [x[0] for x in data]
    labels = [x[1] for x in data]
    sent_ids = [x[2] for x in data]

    encoding = tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])
    labels = torch.LongTensor(labels)

    return token_ids, attention_mask, labels, sents, sent_ids

def single_sentence_collate_fn(all_data):
    token_ids, attention_mask, labels, sents, sent_ids = single_sentence_pad__data(all_data)

    batched_data = {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            #'sents': sents,
            'sent_ids': sent_ids
        }

    return batched_data

"""
This is the implementation for similarity task
We currently use concatenated sentences
"""
def sentence_pair_collate_fn(data, isRegression = True):
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
    if isRegression:
        labels = torch.FloatTensor(labels)
    else:
        labels = torch.LongTensor(labels)
    # the name here should be the same as the forward function you use!!!
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        'sent_ids': sent_ids
    }

# """
# Shift tensor and whatever into device
# """
# def to_device_collator(x):
#     newx = {}
#     for i in x:
#         if hasattr(x[i], "to"):
#             newx[i] = x[i].to(device)
#         else:
#             newx[i] = x[i]
#     return newx

def collate_fn(x):
    task_id, data = x
    # the format should depend on the datasets
    # so we use different ways to collate data
    if task_id == "sentiment_analysis":
        data = single_sentence_collate_fn(data)
    elif task_id == "paraphrase":
        data = sentence_pair_collate_fn(data, isRegression=False)
    elif task_id == "similarity":
        # currently we concatenate two sentences into one
        # maybe we can try different ways to do so
        data = sentence_pair_collate_fn(data, isRegression=True)
    else:
        assert False
    data["task_id"] = task_id
    # huggingface trainer will unpack data and pass this to model's forward function
    return data

# create a training dataset
# evaluation dataset is handled differently
training_dataset = MultiTaskDataset(all_dataset["train"])
order = ["similarity", "sentiment_analysis", "paraphrase", ]
sampler = MultiTaskSequentialBatchSampler(training_dataset, order = order, batch_size=16, drop_last=True)
data_loader = DataLoader(training_dataset, 
                              batch_size=None, 
                              sampler=sampler, 
                              batch_sampler=None, 
                              collate_fn=collate_fn)

# get a feeling of what this data loader returns
next(iter(data_loader))

"""
sanity check, ensure that our collate_fn is implemented correctly
"""
# for i in tqdm(data_loader):
#     pass

class MultitaskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.return_dict = True
        self.bert = BertModel(config)
        # it's important to put it into module dict
        self.lmheaders = nn.ModuleDict({
            # five-way classification
            'sentiment_analysis' : nn.Linear(config.hidden_size, 5),
            # binary classification
            'paraphrase' : nn.Linear(config.hidden_size, 2),
            # regression
            'similarity' : nn.Linear(config.hidden_size, 1)
            })

    def forward(self, input_ids, sent_ids, task_id : str, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False
        )
        
        # Select the embeddings from the specified layer
        sequence_output = outputs.last_hidden_state
        logits = self.lmheaders[task_id](sequence_output[:, 0, :])  # We take the embedding of the first token ([CLS]) for classification.

        # Continue with the loss calculation if labels are provided
        loss = None
        if labels is not None:
            # switch the lm header based on the task id
            if task_id == "sentiment_analysis" or  task_id == 'paraphrase':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif task_id == 'similarity':
                # TODO : actually we should use a activation function?
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels)
            else:
                assert False

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

model = MultitaskBertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda")
model = model.to(device)  # Move the model to the GPU

lr = 5e-06
epochs = 1
batch_size = 16

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    #save_strategy="epoch",
    #load_best_model_at_end=True,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs,
    remove_unused_columns=False,
)

from evaluate import load
glue_metric = load('glue', 'sst2')
compute_metrics_dict = {
    'sentiment_analysis' :  load("accuracy"),
    'paraphrase' : load('glue', 'qqp'),
    'similarity' : load('glue', 'stsb')
}

import numpy as np
def compute_metrics(eval_preds):
    global task_id
    print(task_id)
    metric = compute_metrics_dict[task_id]
    if task_id == 'sentiment_analysis' or task_id == 'paraphrase':
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    elif task_id == "similarity":
        logits, labels = eval_preds
        return metric.compute(predictions=logits, references=labels)
    else:
        assert False

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return data_loader
    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        # hack... we need to know how to process the data here
        global task_id
        task_id = None
        for i in all_dataset["dev"]:
            if all_dataset["dev"][i] == eval_dataset:
                task_id = i
                break
        # inject a task id to input
        def collate_wrapper(f):
            def inner(x):
                d = f(x)
                d["task_id"] = task_id
                return d
            return inner
        if task_id in ["sentiment_analysis"]:
            return DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_wrapper(single_sentence_collate_fn))
        elif task_id in ['paraphrase']:
            return DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_wrapper(
                lambda x : sentence_pair_collate_fn(x, isRegression=False)
                ))
        elif task_id in ['similarity']:
            return DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_wrapper(sentence_pair_collate_fn))
        else:
            assert False

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=None,
    # huggingface trainer supports multiple evaluation dataset!
    eval_dataset=all_dataset["dev"],
    compute_metrics=compute_metrics,
)
trainer.evaluate()
trainer.train()
trainer.evaluate()

