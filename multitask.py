import torch.utils
from transformers import (
    BertModel,
    BertPreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader
from torch import nn
import torch
from typing import Dict, Iterator, List, Tuple, Optional, Any, cast
from tqdm.autonotebook import tqdm
from minbert_datasets import (
    training_dataset_map,
    eval_dataset_map,
    training_dataset,
    MultiTaskSequentialBatchSampler,
    all_task_ids,
)
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


class MultitaskBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.return_dict = True
        self.bert = BertModel(config)
        # it's important to put it into module dict
        self.lmheaders = nn.ModuleDict(
            {
                # five-way classification
                "sentiment_analysis": nn.Linear(config.hidden_size, 5),
                # binary classification
                "paraphrase": nn.Linear(config.hidden_size, 2),
                # regression
                "similarity": nn.Linear(config.hidden_size, 1),
            }
        )

    def forward(
        self,
        input_ids,
        # sent_ids is not used in forward
        sent_ids,
        task_id: str,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
        )

        # Select the embeddings from the specified layer
        sequence_output = outputs.last_hidden_state
        logits = self.lmheaders[task_id](
            sequence_output[:, 0, :]
        )  # We take the embedding of the first token ([CLS]) for classification.

        # Continue with the loss calculation if labels are provided
        loss = None
        if labels is not None:
            # switch the lm header based on the task id
            if task_id == "sentiment_analysis" or task_id == "paraphrase":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            elif task_id == "similarity":
                # TODO : actually we should use a activation function?
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(-1), labels)
            else:
                assert False

        return SequenceClassifierOutput(loss=loss, logits=logits)


model = cast(
    MultitaskBertModel, MultitaskBertModel.from_pretrained("bert-base-uncased")
)
model = model.to(device)  # Move the model to the GPU

# lr = 5e-06
# epochs = 1
# batch_size = 16

from evaluate import load

glue_metric = load("glue", "sst2")
compute_metrics_dict = {
    "sentiment_analysis": load("accuracy"),
    "paraphrase": load("glue", "qqp"),
    "similarity": load("glue", "stsb"),
}

lr = 2e-5
num_epochs = 4
optimizers = {}
lr_schedulers = {}
batch_size = 16
for task_id in all_task_ids:
    optimizers[task_id] = torch.optim.AdamW(model.lmheaders[task_id].parameters(), lr)
    # in epoch, we will go through the each dataset once
    # the steps in each epoch are then calculated as the length of the dataset divided by the batch size
    total_steps = num_epochs * (len(training_dataset.dataset_map[task_id]) // batch_size)
    num_warmup_steps = int(total_steps * 0.1)
    lr_schedulers[task_id] = get_linear_schedule_with_warmup(optimizers[task_id], num_warmup_steps, total_steps)
    # lr_schedulers[task_id] = get_constant_schedule_with_warmup(
    #     optimizers[task_id], num_warmup_steps
    # )

model_optmizer = torch.optim.AdamW(model.bert.parameters(), lr=lr)
model_lr_scheduler = get_constant_schedule_with_warmup(model_optmizer, num_warmup_steps)


def collate_fn(x):
    task_id, data = x
    # the format should depend on the datasets
    # so we use different ways to collate data
    data = training_dataset_map[task_id].collate_fn(data)
    data["task_id"] = task_id
    # huggingface trainer will unpack data and pass this to model's forward function
    return data


# """
# Shift tensor and whatever into device
# """
def to_device_collator(x):
    newx = {}
    for i in x:
        if hasattr(x[i], "to"):
            newx[i] = x[i].to(device)
        else:
            newx[i] = x[i]
    return newx


"""
A brief explanation of what happens here
1. For loop will call `__iter__` method of the data loader
2. The data loader invokes the sampler's `__iter__` method (in this case, MultiTaskSequentialBatchSampler), to get a batch of indices
3. The indices are passed into the dataset's `__getitem__` method (in this case, MultiTaskDataset), to get a batch of data
4. The data are passed into the `collate_fn` function, which is resposible for packing everything into tensor.
"""

order = ["similarity", "sentiment_analysis", "paraphrase"]
"""
Size of dataset :
Similarity is 6000
Paraphrase is 14000
Sentiment is 9000
"""

sampler = MultiTaskSequentialBatchSampler(
    training_dataset, order=order, batch_size=batch_size, drop_last=True
)
data_loader = DataLoader(
    training_dataset,
    batch_size=None,
    sampler=sampler,
    batch_sampler=None,
    collate_fn=collate_fn,
)

# get a feeling of what this data loader returns
next(iter(data_loader))


def get_eval_dataloader(task_id: str) -> DataLoader:
    eval_dataset = eval_dataset_map[task_id]
    batch_size = 16
    def collate_wrapper(f):
        def inner(x):
            d = f(x)
            d["task_id"] = task_id
            return d
        return inner
    collate_fn = eval_dataset.collate_fn
    return DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_wrapper(collate_fn),
    )


def compute_metrics(eval_preds, task_id):
    metric = compute_metrics_dict[task_id]
    if task_id == "sentiment_analysis" or task_id == "paraphrase":
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    elif task_id == "similarity":
        logits, labels = eval_preds
        return metric.compute(predictions=logits, references=labels)
    else:
        assert False


def evaluate():
    evaluate_result = {}
    with torch.no_grad():
        model.eval()
        for task_id in ["sentiment_analysis", "paraphrase", "similarity"]:
            total_loss = 0.0
            logits = []
            labels = []
            loader = get_eval_dataloader(task_id)
            for batch in tqdm(loader):
                batch = to_device_collator(batch)
                output = model(**batch)
                total_loss += output.loss
                logits.append(output.logits)
                labels.append(batch["labels"])
            total_loss /= len(loader)
            logits = torch.cat(logits, dim=0)
            labels = torch.cat(labels, dim=0)
            metric = compute_metrics(
                (logits.detach().cpu().numpy(), labels.detach().cpu().numpy()), task_id
            )
            metric["loss"] = total_loss.item()
            evaluate_result[task_id] = metric
            print(f"{task_id} eval loss: {total_loss}, metric: {metric}")
    return evaluate_result
