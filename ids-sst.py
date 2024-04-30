from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import evaluate
import pandas as pd
import torch
import csv
# from transformers.modeling_bert import 


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoding, self.labels = self.pad_data(dataset)

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, idx):
    #     return self.dataset[idx]
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        # sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        # token_ids = torch.LongTensor(encoding['input_ids'])
        # attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        # return token_ids, attention_mask, labels, sents, sent_ids
        return encoding, labels

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def encode_reviews(reviews, max_length=512):
    return tokenizer(reviews, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data
metric = evaluate.load("accuracy")
device = torch.device("mps")
# model = BertForSequenceClassification.from_pretrained("./test_trainer/checkpoint-1707")

# the layers we're filtering
layers = [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
for i in range(0, 3):
    print(i)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    lr = 5e-06
    epochs = 1
    batch_size = 100
    layer = layers[i]
    print("Learning rate", lr, "Epochs", epochs, "Batch size", batch_size, "Layer", layer)

    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", save_strategy="epoch", 
                                    load_best_model_at_end=True, learning_rate=lr, per_device_train_batch_size=batch_size, num_train_epochs=epochs)


    # Load the dataset
    train_data, num_labels = load_data('data/ids-sst-train.csv')
    test_data = load_data('data/ids-sst-dev.csv', 'val')
    # Dictionary to store value counts
    value_counts = {}

    # Counting unique values in the second index of each tuple
    for item in train_data:
        value = item[1]
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    # Printing value counts
    # for value, count in value_counts.items():
    #     print(f"{value}: {count}")

    train_dataset = SentimentDataset(train_data, {'version': 'train'})
    test_dataset = SentimentDataset(test_data, {'version': 'test'})
    model = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=5) 
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5) 
    # model = BertSentimentClassifier({'hidden_dropout_prob': 0.1,
    #               'num_labels': num_labels,
    #               'hidden_size': 768,
    #               'data_dir': '.',
    #               'option': 'pretrain'})
    model = model.to(device)
    # Create binary labels
    # df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    # X = df['Text'].tolist()
    # y = df['label'].values

    for name, param in model.named_parameters():
        if "layer." in name:
            found = False
            for l in layer:
                if "." + str(l) + "." in name:
                    found = True
            if found:
                param.requires_grad = True
            else:
                print(name)
                param.requires_grad = False
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train_encodings = encode_reviews(X_train)
    # test_encodings = encode_reviews(X_test)
    # train_dataset = ReviewsDataset(train_encodings, y_train)
    # test_dataset = ReviewsDataset(test_encodings, y_test)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        # data_collator=train_dataset.collate_fn
    )

    trainer.train()
    print(trainer.evaluate())
    # model_save_path = 'BertFT'
    # trainer.save_model(model_save_path)
