import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from torch.utils.tensorboard import SummaryWriter
from transformers import logging as transformers_logging
from customModel import *
# 1. Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load and observe the dataset
df = pd.read_csv('quora-train.csv', sep='\t')
df = df.sample(10000)
class_0 = df[df['is_duplicate'] == 0]
class_1 = df[df['is_duplicate'] == 1]
sampleNum = 1500
# Sample from each class
# class_0_sample = class_0.sample(sampleNum, random_state=42)
# class_1_sample = class_1.sample(sampleNum, random_state=42)
# df = pd.concat([class_0_sample, class_1_sample])
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
if df['is_duplicate'].isnull().any():
    # Remove rows with NaN values in 'is_duplicate'
    df = df.dropna(subset=['is_duplicate'])
df['is_duplicate'] = df['is_duplicate'].astype(int)  # Convert labels from float to int
# df_test = pd.read_csv('quora-test-student.csv', sep='\t')
# first_10_rows = df.head(10) 
# print(first_10_rows)
# 2. Observe dataset distribution
# print("Sentence length distribution:")
# print(df['sentence1'].str.len().describe())
# print(df['sentence2'].str.len().describe())

# print("Label distribution (0 - not duplicate, 1 - duplicate):")
# print(df['is_duplicate'].value_counts())

# 3. Preprocess data
# def clean_text(text):
#     if isinstance(text, str):  # Check if text is a string
#         text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#         text = re.sub(r'[^\w\s]', '', text)  # Remove weird symbols
#     else:
#         text = ''  # If text is not a string, replace it with empty string
#     return text

# df['sentence1'] = df['sentence1'].apply(clean_text)
# df['sentence2'] = df['sentence2'].apply(clean_text)

X = df[['sentence1', 'sentence2']]
y = df['is_duplicate']             

# Split the dataset into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['sentence1'] = X_train['sentence1'].astype(str)
X_train['sentence2'] = X_train['sentence2'].astype(str)
X_val['sentence1'] = X_val['sentence1'].astype(str)
X_val['sentence2'] = X_val['sentence2'].astype(str)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the sentence pairs
train_encodings = tokenizer(X_train['sentence1'].tolist(), X_train['sentence2'].tolist(), truncation=True, padding='max_length', max_length=512)
val_encodings = tokenizer(X_val['sentence1'].tolist(), X_val['sentence2'].tolist(), truncation=True, padding='max_length', max_length=512)

# decoded = tokenizer.decode(train_encodings["input_ids"][0])
# print(decoded[0])

def encode_questions(questions1, questions2):
    # Ensure that both questions1 and questions2 are lists of strings
    if isinstance(questions1, (list, tuple)) and isinstance(questions2, (list, tuple)):
        return tokenizer(questions1, questions2, padding=True, truncation=True, return_tensors="pt")
    else:
        raise ValueError("Questions must be lists or tuples of strings")

# Create torch dataset
class QuoraDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, layer_idx=-1):
        self.encodings = encodings
        # Convert labels to a list if they are in a pandas Series
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)
        self.layer_idx = layer_idx

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



train_dataset = QuoraDataset(train_encodings, y_train)
val_dataset = QuoraDataset(val_encodings, y_val)
batch_size = 16
learning_rate = 2e-5
epoch = 3
output_layer = 11
out_dir = f"test_trainer_embedding_{batch_size}_{output_layer}"

# Load BERT model
model = CustomBertModel.from_pretrained('bert-base-uncased', num_labels=2, layer_num=output_layer)
model.cuda()  # If using GPU


# Training arguments and hyperparameter selection
training_args = TrainingArguments(
    output_dir=out_dir,
    evaluation_strategy="steps",
    logging_dir="./logs",  # Directory for TensorBoard logs
    logging_steps=50,  # Log every 50 steps
    per_device_train_batch_size=batch_size,  # Hyperparameter: Batch size
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,  # Hyperparameter: Learning rate
    num_train_epochs=epoch,  # Hyperparameter: Epoch
    load_best_model_at_end=True,
    weight_decay=0.01,
)

# Initialize TensorBoard
writer = SummaryWriter(log_dir="./logs")

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits = eval_pred.predictions  # Access logits directly
    labels = eval_pred.label_ids  # Make sure that labels are passed correctly as label_ids
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate()
bert_name = f"Bert_Embedding_{output_layer}"
# Save the model
model_save_path = bert_name
trainer.save_model(model_save_path)

# Close TensorBoard writer
writer.close()
