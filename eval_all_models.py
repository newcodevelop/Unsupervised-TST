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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os
device = "cuda"
# model_id = "gpt2-medium"
model_id='./gpt2_lm/checkpoint-9600'
gpt2 = GPT2LMHeadModel.from_pretrained(model_id).to(device)
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")

import numpy as np
def get_ppl(sent):

    encodings = gpt2_tokenizer(sent, return_tensors="pt")



    max_length = gpt2.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = gpt2(input_ids, labels=target_ids)
            # print(outputs[0])
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    # print(ppl, (torch.stack(nlls).sum() / end_loc))
    if ppl.item()==np.nan:
        ppl = torch.tensor(10)

    return ppl.item()

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




from tqdm import tqdm

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

dd = [{'src': train_src[i], 'tgt': train_tgt[i], 'lab': train_lab[i]} for i in range(len(train_src))][0:443232]
dd_t = [{'src': test_src[i], 'tgt': test_tgt[i], 'lab': test_lab[i]} for i in range(len(test_src))][0:1000]



import json
with open("./data.json", 'w') as f:
    for item in dd:
        f.write(json.dumps(item) + "\n")

import json
with open("./data_dev.json", 'w') as f:
    for item in dd_t:
        f.write(json.dumps(item) + "\n")

from torchtext.legacy import data
from torchtext.legacy import datasets

LAB = data.Field(sequential=False, use_vocab=False)


SRC = Field(tokenize =tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            # fix_length=100,
            batch_first = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            # fix_length=100,
            batch_first = True)

fields = {'src': ('s', SRC), 'tgt': ('t', TRG), 'lab': ('l', LAB)}

train_data,test_data  = data.TabularDataset.splits(
                            path = '.',
                            train = 'data.json',
                            test = 'data_dev.json',
                            format = 'json',
                            fields = fields
)

print(test_data)


SRC.build_vocab(train_data, min_freq = 1)
TRG.build_vocab(train_data, min_freq = 1)

BATCH_SIZE = 32

train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data), 
    sort_key = lambda x: len(x.s),
     batch_size = BATCH_SIZE,
     device = device)


print(len(test_iterator))





train_texts = [i['src'] for i in dd]
train_labels = [i['lab'] for i in dd]

test_texts = [i['src'] for i in dd_t]
test_labels = [i['lab'] for i in dd_t]


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)


print(train_texts[0:10])
print(val_texts[0:10])
print(train_labels[0:10])
print(val_labels[0:10])

T = ['__label__'+str(train_labels[i])+' '+train_texts[i] for i in range(len(train_labels))]
V = ['__label__'+str(val_labels[i])+' '+val_texts[i] for i in range(len(val_labels))]


print(T[0:10])
print(V[0:10])

with open('train_fasttext.txt', 'w') as f:
    for line in T:
        f.write(f"{line}\n")

with open('valid_fasttext.txt', 'w') as f:
    for line in V:
        f.write(f"{line}\n")

#follow https://fasttext.cc/docs/en/supervised-tutorial.html

import fasttext
model = fasttext.train_supervised(input="train_fasttext.txt",  wordNgrams=1)
model.save_model("model_fasttext.bin")
print(model.test("valid_fasttext.txt"))




t2l = {}

for txt,lab in zip(test_src,test_lab):
    t2l[txt]=lab

print(t2l)






# print(src_txt[0:2])
# print(labs[0:2])
# print(predicted_txt[0:2])

from mosestokenizer import *


from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertForSequenceClassification.from_pretrained('./results_yelp')

def get_pred(labs,predicted_txt,source_txt):
    style_labs_fasttext = []
    style_labs = []
    for i in predicted_txt:
        inputs = tokenizer(i, return_tensors="pt")

        with torch.no_grad():
          logits = distilbert(**inputs, return_dict=True).logits

        pred_style = logits.argmax().item()
        style_labs.append(pred_style)
        style_labs_fasttext.append(int(model.predict(i)[0][0][-1]))

    from sklearn.metrics import f1_score,accuracy_score
    print(f1_score(labs,style_labs,average='macro'))
    print(f1_score(labs,style_labs,average='weighted'))
    print(accuracy_score(labs,style_labs))


    print(f1_score(labs,style_labs_fasttext,average='macro'))
    print(f1_score(labs,style_labs_fasttext,average='weighted'))
    print(accuracy_score(labs,style_labs_fasttext))

    return accuracy_score(labs,style_labs)

print(model.predict("Why not put knives in the dishwasher?")[0][0][-1])
print(model.predict("This was not so great"))
print(model.predict("The food was awesome, the service was shitty"))
print(model.predict("I loved it!!"))
import evaluate
import pandas as pd
meteor = evaluate.load('meteor')
bertscore = evaluate.load('bertscore')
from sacrebleu.metrics import BLEU, CHRF, TER

bleu = BLEU()



import pandas as pd
df = pd.read_csv('results_yelp_nll_eval.csv')
print(df.keys())
predicted_txt,src_txt,references,sent_for_ppl,test_labs = list(df['target']), list(df['source']), list(df['reference']), list(df['sent_for_ppl']), list(df['label'])


labs = list(np.array(test_labs)^1)
print('Ours')
acc = get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in sent_for_ppl:
    # print(i,i[:-1])
    # exit(0)
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
print('***********************************')




print('*********************************')
src_txt,predicted_txt,labs = [],[],[]
references = []
dir_path = './Direct-Style-Transfer/output'
with open(dir_path+'/yelp.direct.0.txt', 'r') as f:
  k_0 = f.readlines()
  for i in k_0:
    src,ref,tgt = i.split('\t')[0].strip(),i.split('\t')[1].strip(),i.split('\t')[2].strip()
    src_txt.append(src)
    predicted_txt.append(tgt)
    labs.append(0)
    references.append(ref)

with open(dir_path+'/yelp.direct.1.txt', 'r') as f:
  k_0 = f.readlines()
  for i in k_0:
    src,ref,tgt = i.split('\t')[0].strip(),i.split('\t')[1].strip(),i.split('\t')[2].strip()
    src_txt.append(src)
    predicted_txt.append(tgt)
    labs.append(1)
    references.append(ref)


src_txt=[i[:-1].strip() for i in src_txt]
predicted_txt = [i[:-1].strip() for i in predicted_txt]
references = [i[:-1].strip() for i in references]
print(src_txt[0:2])
print(predicted_txt[0:2])
print(references[0:2])
# exit(0)
labs = list(np.array(labs)^1)
print('DIR')
acc = get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)


print('***********************************')
# exit(0)
src_txt,predicted_txt,labs = [],[],[]
df_0 = pd.read_csv('./transformer-drg-style-transfer/results/yelp/yelp_all_model_prediction_ref0.csv')
df_1 = pd.read_csv('./transformer-drg-style-transfer/results/yelp/yelp_all_model_prediction_ref1.csv')
predicted_txt = list(df_0['BERT_DEL'])+list(df_1['BERT_DEL'])
src_txt = list(df_0['Source'])+list(df_1['Source'])
labs = [0]*len(df_0)+[1]*len(df_1)
labs = list(np.array(labs)^1)
print('B-GST')
acc=get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
src_txt,predicted_txt,labs = src_txt,[],[]
with open('./style-transformer/outputs/yelp/output.0.ours_cond','r') as f:
    k0 = f.readlines()
    k0 = list(map(lambda x: x.strip(), k0))

with open('./style-transformer/outputs/yelp/output.1.ours_cond','r') as f:
    k1 = f.readlines()
    k1 = list(map(lambda x: x.strip(), k1))

predicted_txt=k0+k1
labs = [0]*len(k0)+[1]*len(k1)
labs = list(np.array(labs)^1)
print('style-transformer')
acc=get_pred(labs,predicted_txt,src_txt)
# print(len(predicted_txt),len(src_txt),print(references))
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
src_txt,predicted_txt,labs = [],[],[]
with open('deep_latent_yelp.txt', 'r') as f:
  k_0 = f.readlines()
  for i in k_0:
    src,tgt = i.split('\t')[0].strip(),i.split('\t')[1].strip()
    src_txt.append(src)
    predicted_txt.append(tgt)
labs = [0]*500+[1]*500
labs = list(np.array(labs)^1)
print('Deep Latent')
acc=get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
src_txt,predicted_txt,labs = src_txt,[],labs

with open('./DualRL/outputs/yelp/DualRL/test.0.tsf','r') as f:
    k0 = f.readlines()
    k0 = list(map(lambda x: x.strip(), k0))

with open('./DualRL/outputs/yelp/DualRL/test.1.tsf','r') as f:
    k1 = f.readlines()
    k1 = list(map(lambda x: x.strip(), k1))
predicted_txt=k0+k1
print('DualRL')
acc=get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
src_txt,predicted_txt,labs = src_txt,[],labs

with open('./Sentiment-and-Style-Transfer/evaluation/outputs/yelp/sentiment.test.0.orgin','r') as f:
    k0 = f.readlines()
    k0 = list(map(lambda x: x.strip(), k0))

with open('./Sentiment-and-Style-Transfer/evaluation/outputs/yelp/sentiment.test.1.orgin','r') as f:
    k1 = f.readlines()
    k1 = list(map(lambda x: x.strip(), k1))
predicted_txt=k0+k1
print('DeleteAndRetrieve')
acc=get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')
src_txt,predicted_txt,labs = src_txt,[],labs

with open('./yelp_output_0.txt','r') as f:
    k0 = f.readlines()
    k0 = list(map(lambda x: x.strip(), k0))

with open('./yelp_output_1.txt','r') as f:
    k1 = f.readlines()
    k1 = list(map(lambda x: x.strip(), k1))
predicted_txt=k0+k1
print('Tag&Gen')
acc=get_pred(labs,predicted_txt,src_txt)
b1 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=src_txt, model_type="distilbert-base-uncased")['f1']))
b2 = np.mean(np.array(bertscore.compute(predictions=predicted_txt, references=references, model_type="distilbert-base-uncased")['f1']))
print(b1)
print(b2)
msrc = meteor.compute(predictions=predicted_txt, references=src_txt)
mref = meteor.compute(predictions=predicted_txt, references=references)
bsrc = bleu.corpus_score(predicted_txt, [src_txt])
bref = bleu.corpus_score(predicted_txt, [references])
print(msrc,mref,bsrc,bref)
PPL = []
cnt= 0
for i in predicted_txt:
    ppl = get_ppl(i)
    PPL.append(ppl)
    cnt+=1
PPL = np.array(PPL)
PPL = np.nan_to_num(PPL, copy=True, nan=0.0)
print(np.mean(PPL))
jsrc = (acc*b1*msrc['meteor']*(bsrc.score/100)*(1/np.mean(PPL)))**(1/5)
# print(jsrc)
jref = (acc*b2*mref['meteor']*(bref.score/100)*(1/np.mean(PPL)))**(1/5)
print(jsrc,jref)
print('***********************************')

