import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchtext
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import os
device = 'cuda'
import en_core_web_sm, de_core_news_sm
spacy_en = en_core_web_sm.load()

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]




dir_path = './Sentiment-and-Style-Transfer/data/yelp/'
with open(dir_path+'./sentiment.train.0', 'r') as f:
  k_0 = f.readlines()
k_0 = list(map(lambda x: x.strip(), k_0))

with open(dir_path+'./sentiment.train.1', 'r') as f:
  k_1 = f.readlines()
k_1 = list(map(lambda x: x.strip(), k_1))

train_src = k_0+k_1
train_lab = [0]*len(k_0)+[1]*len(k_1)
train_tgt = train_src

with open(dir_path+'./sentiment.test.0', 'r') as f:
  k_0 = f.readlines()
k_0 = list(map(lambda x: x.strip(), k_0))

with open(dir_path+'./sentiment.test.1', 'r') as f:
  k_1 = f.readlines()
k_1 = list(map(lambda x: x.strip(), k_1))

test_src = k_0+k_1
test_lab = [0]*len(k_0)+[1]*len(k_1)
test_tgt = test_src

dd = [{'src': train_src[i], 'tgt': train_tgt[i], 'lab': train_lab[i]} for i in range(len(train_src))]
dd_t = [{'src': test_src[i], 'tgt': test_tgt[i], 'lab': test_lab[i]} for i in range(len(test_src))]



train_texts = [i['src'] for i in dd]
train_labels = [i['lab'] for i in dd]

test_texts = [i['src'] for i in dd_t]
test_labels = [i['lab'] for i in dd_t]


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)



from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir='./results_yelp',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs_yelp',            # directory for storing logs
    logging_steps=10,
    seed=0,
    save_total_limit = 1,
    load_best_model_at_end=True,
    save_strategy = "no"
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
# trainer.save_model('./results')
trainer.save_model('./results_yelp')


# model = DistilBertForSequenceClassification.from_pretrained('./results')

model = DistilBertForSequenceClassification.from_pretrained('./results_yelp')

# arguments for Trainer
test_args = TrainingArguments(
    output_dir = './results_infer_yelp',
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,   
    # dataloader_drop_last = False    
)

# init trainer
trainer = Trainer(
              model = model, 
              args = test_args, 
              # compute_metrics = compute_metrics
              )

test_results = trainer.predict(test_dataset)
print(test_results)

preds = np.argmax(test_results.predictions,axis=-1)

labels = test_results.label_ids
from sklearn.metrics import accuracy_score,f1_score
print(accuracy_score(labels,preds))
print(f1_score(labels,preds))

