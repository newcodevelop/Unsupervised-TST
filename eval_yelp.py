from email.mime import base
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchtext
from torchtext.legacy.data import Field, BucketIterator
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.distributions import Categorical
import spacy
import numpy as np
import torch.nn.functional as F
import random
import math
import time
import resource
from sacrebleu import sentence_bleu
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import tensorflow as tf
# https://ai.stackexchange.com/questions/2405/how-do-i-handle-negative-rewards-in-policy-gradients-with-the-cross-entropy-loss

import tensorflow_probability as tfp_
tfd = tfp_.distributions
tfkl = tf.keras.layers
tfp = tfp_.layers











device = 'cuda'
import en_core_web_sm, de_core_news_sm
spacy_en = en_core_web_sm.load()






def tokenize_de(text,max_length=100):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][:max_length-2]

def tokenize_en(text,max_length=100):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)][:max_length-2]


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


from transformers import DistilBertTokenizerFast,DistilBertForSequenceClassification
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertForSequenceClassification.from_pretrained('./results_yelp')






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

BATCH_SIZE = 50

train_iterator,test_iterator = BucketIterator.splits(
    (train_data,test_data), 
    sort_key = lambda x: len(x.s),
     batch_size = BATCH_SIZE,
     device = device)


print(len(train_iterator))

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention



class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 20):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.ffn = nn.Linear(hid_dim,2)
        
    def forward(self, src, src_mask, is_discriminator=False, if_req=False):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        # print(torch.max(src),torch.min(src))
        # print(src_mask.shape)
        # print('tok embedding shape {}'.format(self.tok_embedding.weight.repeat((8,1,1)).shape))
        batch_size = src_mask.shape[0]
        src_len = src_mask.shape[3]
        # src_len = min(src_len,100)
        # src = src[:,0:src_len]
        # src_mask = src_mask[:,:,:,0:src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        if if_req:
          # print('in if_req')
          # print(src.shape)
          # print(self.tok_embedding.weight.repeat((8,1,1)).shape)
          
          src_ = self.dropout( (torch.bmm(src,self.tok_embedding.weight.repeat((8,1,1)))*self.scale) + self.pos_embedding(pos) )
        else:
          src_ = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        src__ = self.layers[0](src_,src_mask)
        for layer in self.layers[1:]:
            src__ = layer(src__, src_mask)
            
        #src = [batch size, src len, hid dim]
        
        if if_req:
          return src__,src_


        if not is_discriminator:
          return src__
        else:
          return self.ffn(src__.mean(dim=1))


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
        # print(torch.max(src))
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return torch.tensor(inp.long(),requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        print(F.hardtanh(grad_output))
        return F.hardtanh(grad_output)
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 20):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.max_length = max_length
        self.ste = StraightThroughEstimator()
        
    def forward(self, trg, enc_src, trg_mask, src_mask, is_policy=False, gumbel_softmax=False):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # print(trg_len)
        # trg_len = min(trg_len,100)
        # trg = trg[:,0:trg_len]
        # trg_mask= trg_mask[:,:,0:trg_len,0:trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)

        if is_policy:
            # pred_token = output.argmax(2)[:,-1].item()
            # output_shape = (bs,length,hidden_dim)
            output = output[:,-1,:] #select output token from last state
            token_probs = F.softmax(output,dim=1) # softmax prob distribution
            m = Categorical(token_probs)
            action = m.sample()
            
            log_prob = m.log_prob(action)
            return action, log_prob
        # https://www.hassanaskary.com/python/pytorch/deep%20learning/2020/09/19/intuitive-explanation-of-straight-through-estimators.html
        if gumbel_softmax:
          gumbel_one_hot = F.gumbel_softmax(output[:,-1,:].squeeze(),tau=1,hard=True)
          # print(gumbel_one_hot)
          # print(gumbel_one_hot.size())
          ll = int(list(gumbel_one_hot.size())[1])
          aranged = torch.arange(ll).repeat(8,1).to('cuda')
          # print(aranged,aranged.shape)

          hard_sample_w_grad = torch.sum(aranged*gumbel_one_hot,axis=1) # reparameterization to get one hot vector
          hard_sample_w_grad = hard_sample_w_grad.unsqueeze(0)
          # hard_sample_wo_grad = self.ste(hard_sample_w_grad)
          hard_sample_wo_grad = hard_sample_w_grad.long()

          return output, attention, hard_sample_wo_grad, gumbel_one_hot



        
        #output = [batch size, trg len, output dim]
            
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention



INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 256
DEC_PF_DIM = 256
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
print(INPUT_DIM)
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder_humor,
                 decoder_nonhumor, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder_humor = decoder_humor
        self.decoder_nonhumor = decoder_nonhumor
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.style_proj_h = nn.Linear(HID_DIM,HID_DIM)
        self.style_proj_nh = nn.Linear(HID_DIM,HID_DIM)
        self.content_proj = nn.Linear(HID_DIM,HID_DIM)
        

        self.style_labeler = nn.Linear(HID_DIM,2)
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)

        #divide enc_src to two separate vectors having same dimension as enc_src but should encode style and contents respectively

        style_h = self.style_proj_h(enc_src)
        style_nh = self.style_proj_nh(enc_src)

        # style_h_ = torch.tensor(style_h.detach().cpu().numpy()).to('cuda')
        # style_nh_ = torch.tensor(style_nh.detach().cpu().numpy()).to('cuda')

        # for estimator, mi_params in estimators.items():
        #     # print("Training %s..." % estimator)
        #     w1,w2,b1,b2 = train_estimator_torch(critic_params, data_params, mi_params, style_h_, style_nh_)
        #  # = train_estimator_torch(critic_params, data_params, mi_params, style, content)



        # w1 = torch.tensor(w1.detach().cpu().numpy()).to('cuda')
        # w2 = torch.tensor(w2.detach().cpu().numpy()).to('cuda')
        # b1 = torch.tensor(b1.detach().cpu().numpy()).to('cuda')
        # b2 = torch.tensor(b2.detach().cpu().numpy()).to('cuda')

        # estimates = []
        # # print(w1,w2,b1,b2)
        # x,y = style_h,style_nh
        # ll = x.size()[1]
        # for i in range(ll):


        #   # w1,w2,b1,b2 = train_step(data_params, mi_params,x[:,i,:].detach().clone(),y[:,i,:].detach().clone())
          
        #   batch_size = x[:,i,:].size()[0]
        #   x_t_tiled = torch.tile(x[:,i,:].unsqueeze(dim=0),(batch_size,1,1))
        #   y_t_tiled = torch.tile(y[:,i,:].unsqueeze(dim=1),(1,batch_size,1))
        #   xy_pairs = torch.reshape(torch.cat((x_t_tiled, y_t_tiled), dim=2), (batch_size * batch_size, -1))
        #   # print(xy_pairs.size(),w1.size())
        #   k = relu(xy_pairs.mm(w1)+b1)
        #   # print(k.size(),w2.size())
        #   scores = k.mm(torch.t(w2))+b2
          
        #   k_ = torch.t(torch.reshape(scores, (batch_size, batch_size)))
        #   # print(k_)
        #   # print(k_.size())
        #   mi_updated = nwj_lower_bound_torch(k_)
        #   # mi_updated.backward()
        #   # print(mi_updated)
        #   # print(mi_updated.size())
        #   estimates.append(mi_updated)

        # estimates = torch.stack(estimates)

        # mi_alpha = torch.mean(estimates)

        pred_style = self.style_labeler(torch.mean(style_h,dim=1))

        content = self.content_proj(enc_src)

        #style and content's resultant should reconstruct input sentence,so,
        enc_out_h = style_h+content
        enc_out_nh = style_nh+content
        
        #enc_src = [batch size, src len, hid dim]

        
                
        output_h, attention_h = self.decoder_humor(trg, enc_out_h, trg_mask, src_mask)
        output_n, attention_n = self.decoder_nonhumor(trg, enc_out_nh, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        mi_alpha=0.0 #dummy
        return output_h, attention_h, output_n,attention_n, style_h,style_nh, content, pred_style, mi_alpha




enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

disc_nh = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

disc_h = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)


dec_h = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)


dec_n = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)




SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec_h,dec_n, SRC_PAD_IDX, TRG_PAD_IDX, device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)


def flip(p):
    return 'H' if random.random() < p else 'T'
from evaluate import load
# bleurt = load("bleurt", module_type="metric")
def get_reward(a,b,lab, j= False):
   

    p = []
    for i in a.split(' '):
      if i== '<eos>':
        break
      else:
        p.append(i)

    a = ' '.join(p)

    # ble = bleurt.compute(predictions=[a], references=[b])['scores'][0]
    ble = (sentence_bleu(a,[b]).score)/100
    # predictions, raw_outputs = roberta.predict([a])
    # pred_style = predictions[0]
    inputs = tokenizer(a, return_tensors="pt")

    with torch.no_grad():
        logits = distilbert(**inputs).logits

    pred_style = logits.argmax().item()
   

    if j:
        print('Greedy : {}'.format(a))
    else:
        print('Sampled : {}'.format(a))

    print('Reference : {}'.format(b))
    print('BLEURT: {}'.format(ble))
    print('pred style: {}'.format(pred_style))
    print('label : {}'.format(lab))
    if lab==0:
        print('training pos decoder via RL')
    elif lab==1:
        print('training neg decoder via RL')
    print('**************************************')
    if lab==0:
      #trainining humorous decoder only
      return (ble+pred_style)/2
    elif lab==1:
      #training non humorous decoder only
      if pred_style==0:
        return (1+ble)/2
      else:
        return ble/2
    




from tqdm import tqdm
REWARDS = [0]
CCC = [0]
TURN=0
def reinforce(list_of_src_tokens,list_of_trg_tokens, src_field, trg_field, model, device,lab_, max_len = 18):
    

    n_training_episodes=8
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)

    scores_deque = deque(maxlen=10)
    scores = []

    losses = []

    model.train()
    tot_loss=0  
    lot = []
    for i_episode in tqdm(range(n_training_episodes)):
      s = []
      for i in list_of_trg_tokens[i_episode,:]:
        if TRG.vocab.itos[i.item()]=='<pad>':
          break
        s.append(TRG.vocab.itos[i.item()])

           
            
      trg_sentence = ' '.join(s[1:-1])
      lot.append(trg_sentence)
    # print(lot)

    # print(list_of_src_tokens)
    src_tensor = list_of_src_tokens
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content
    # print('enc out h {}'.format(enc_out_h.size()))
    trg_indexes_h,trg_indexes_nh = [],[]

    trg_indexes_h.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*8))
    trg_indexes_nh.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*8))
    pred_toks_h = []
    pred_toks_nh = []
    W_h,W_nh = [],[]
    for i in range(max_len):
        
        trg_tensor_h = torch.LongTensor(trg_indexes_h).T.to(device)
        trg_tensor_nh = torch.LongTensor(trg_indexes_nh).T.to(device)
        # print(trg_tensor_h,trg_tensor_h.shape)
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_h = model.make_trg_mask(trg_tensor_h)

        
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_nh = model.make_trg_mask(trg_tensor_nh)
        
        # print(trg_tensor,trg_mask)
        # exit(0)
        # with torch.no_grad():
        output_h, attention_h, sample_h, w_grad_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask,gumbel_softmax=True)
        output_nh, attention_nh, sample_nh, w_grad_nh = model.decoder_nonhumor(trg_tensor_nh, enc_out_nh, trg_mask_nh, src_mask,gumbel_softmax=True)
        # print(output_h.shape,w_grad_h.shape)
        # print()
        # print('**********')
        # print(output_nh.shape,w_grad_nh.shape)
        
        
        
    
        # w_grad = w_grad.unsqueeze(0)

        pred_token_h = output_h.argmax(2)[:,-1].detach().cpu().numpy()
        pred_token_nh = output_nh.argmax(2)[:,-1].detach().cpu().numpy()
        # print(pred_token)
        # exit(0)
        # print(sample)
        # print(hard_sample_w_grad,pred_token)
        # pred_toks.append(hard_sample_w_grad.unsqueeze(0))
        # pred_toks.append(sample)
        W_h.append(w_grad_h)
        W_nh.append(w_grad_nh)
        pred_toks_h.append(sample_h)
        pred_toks_nh.append(sample_nh)
        # print(sample,w_grad)
        trg_indexes_h.append(pred_token_h)
        trg_indexes_nh.append(pred_token_nh)
        # print(trg_indexes_h)

        
    
    
    # print('((()()()(()()()()()()()()()())))))')
    # trg_tokens_h = [trg_field.vocab.itos[i] for i in trg_indexes_h]
    # trg_tokens_nh = [trg_field.vocab.itos[i] for i in trg_indexes_nh]

    # pred_toks = torch.cat(pred_toks).unsqueeze(0)

    W_h = torch.stack(W_h)
    W_nh = torch.stack(W_nh)
    trg_indexes_h= np.asarray(trg_indexes_h)[1:,:]
    trg_indexes_nh= np.asarray(trg_indexes_nh)[1:,:]
    # print(np.shape(trg_indexes_h))
    # print(np.shape(trg_indexes_nh))
    # print(W_h.shape)
    # print(W_nh.shape)

    W = []
    pred_toks = []
    target = []
    

    # print(pred_toks)
    # pred_toks = pred_toks.type(torch.LongTensor)
    # print(pred_toks)
    # return trg_tokens[1:], attention, pred_toks, W



    pred_toks_h = torch.cat(pred_toks_h)
    pred_toks_nh = torch.cat(pred_toks_nh)
    # print(pred_toks_h.shape,pred_toks_nh.shape)
    # W_h = torch.cat(W_h)
    # W_nh = torch.cat(W_nh)

    # print(W_h.shape,W_nh.shape)
    for i_,i in enumerate(lab_):
      # print(i.item())
      if i.item()==0:
        W.append(W_h[:,i_,:])
        pred_toks.append(pred_toks_h[:,i_])
        target.append(trg_indexes_h[:,i_])
      else:
        W.append(W_nh[:,i_,:])
        pred_toks.append(pred_toks_nh[:,i_])
        target.append(trg_indexes_nh[:,i_])
    
    W = torch.stack(W)
    pred_toks = torch.stack(pred_toks)
    # target = torch.LongTensor(target)

    # print(W.shape,pred_toks.shape,np.shape(target))

    # baseline  []
    baseline = [[trg_field.vocab.itos[i] for i in trg] for trg in target]

    baseline = list(map(lambda i: ' '.join(i[:]),baseline))

    # print(baseline)
    bs_toks_soft = W
    bs_toks = pred_toks



    # print('*****************************************************************************************************')
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content
    # print('enc out h {}'.format(enc_out_h.size()))
    trg_indexes_h,trg_indexes_nh = [],[]

    trg_indexes_h.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*8))
    trg_indexes_nh.append(np.asarray([trg_field.vocab.stoi[trg_field.init_token]]*8))
    logprob_h = []
    logprob_nh = []
    for i in range(max_len):
        
        trg_tensor_h = torch.LongTensor(trg_indexes_h).T.to(device)
        trg_tensor_nh = torch.LongTensor(trg_indexes_nh).T.to(device)
        # print(trg_tensor_h,trg_tensor_h.shape)
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_h = model.make_trg_mask(trg_tensor_h)

        
        # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask_nh = model.make_trg_mask(trg_tensor_nh)
        
        # print(trg_tensor,trg_mask)
        # exit(0)
        # with torch.no_grad():
        # output_h, attention_h, sample_h, w_grad_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask,gumbel_softmax=True)
        action_h, log_prob_h = model.decoder_humor(trg_tensor_h, enc_out_h, trg_mask_h, src_mask, is_policy=True)
        action_nh, log_prob_nh = model.decoder_nonhumor(trg_tensor_nh, enc_out_nh, trg_mask_nh, src_mask, is_policy=True)
        trg_indexes_h.append(action_h.detach().cpu().numpy())
        trg_indexes_nh.append(action_nh.detach().cpu().numpy())
        logprob_h.append(log_prob_h)
        logprob_nh.append(log_prob_nh)

    logprob_h = torch.stack(logprob_h)
    logprob_nh = torch.stack(logprob_nh)
    trg_indexes_h = np.asarray(trg_indexes_h)[1:,:]
    trg_indexes_nh = np.asarray(trg_indexes_nh)[1:,:]
    # print(logprob_h.shape,np.shape(trg_indexes_h))

    logprob = []
    action = []
    for i_,i in enumerate(lab_):
      # print(i.item())
      if i.item()==0:

        logprob.append(logprob_h[:,i_])
        action.append(trg_indexes_h[:,i_])
      else:
        
        logprob.append(logprob_nh[:,i_])
        action.append(trg_indexes_nh[:,i_])
    
    
    logprob = torch.stack(logprob)
    action = np.asarray(action)
    # print(action)
    action_mask = []
    eos_tok = trg_field.vocab.stoi[trg_field.eos_token]
    for i in action:
      try:
        p = [1]*list(i).index(eos_tok)+[0]*(len(list(i))-list(i).index(eos_tok))
      except:
        p = [1]*len(list(i))
      action_mask.append(p)
    action_mask = np.asarray(action_mask)
    # print(action)
    # print(action_mask)


    print(logprob.shape, np.shape(action_mask))

    final_sen = [[trg_field.vocab.itos[i] for i in trg] for trg in action]

    final_sen = list(map(lambda i: ' '.join(i[:]),final_sen))
    # print(lot, baseline, final_sen)
    # print(len(lot), len(baseline), len(final_sen))
    batch_baseline_reward = []
    batch_sample_reward = []
    for i in range(8):
      baseline_reward = get_reward(baseline[i],lot[i],lab_[i].item(),j=True)
      sample_reward = get_reward(final_sen[i],lot[i],lab_[i].item(),j=False)
      batch_baseline_reward.append(baseline_reward)
      batch_sample_reward.append(sample_reward)
    # print(batch_sample_reward)
    # print(batch_baseline_reward)

    R = np.tile(np.asarray(batch_baseline_reward)-np.asarray(batch_sample_reward), (max_len,1)).T

    R = R*action_mask # mask tokens obtained after EOS token by 0 to not assign any reward to them.
    print(R)
    print(np.shape(R),logprob.shape)
    # print(R)
    REWARDS.extend(list(np.asarray(batch_baseline_reward)))

    rl_loss = torch.mean(torch.sum(logprob*torch.tensor(R).to('cuda'),axis=1))
    # print(rl_loss)

    baseline_mask = model.make_src_mask(bs_toks)
    # print(baseline_mask)
    
    # with torch.no_grad():
    enc_baseline, enc_grad = model.encoder(bs_toks_soft, baseline_mask, if_req=True)
    style_h = model.style_proj_h(enc_baseline)

    style_nh = model.style_proj_nh(enc_baseline)

    

    content = model.content_proj(enc_baseline)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

    # print('here')
    
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    # print(enc_out_h.shape, enc_out_nh.shape)
    # print(src_tensor)
    trg_tensor=src_tensor[:,:-1]
    trg_mask = model.make_src_mask(trg_tensor)
    # print(trg_tensor,trg_mask)
    
    op_nh,attn_nh = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, baseline_mask)
    
    op_h,attn_h = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, baseline_mask)

    # print(op_h.shape)

    op_h = op_h.contiguous().view(-1, op_h.shape[-1])
    op_nh = op_nh.contiguous().view(-1, op_nh.shape[-1])

    # print(op_h.shape)

    trg_ = src_tensor[:,1:].contiguous().view(-1)

    # print(op,trg_)
    # trg_ = src_tensor[:,1:]
    # print(trg_.shape)
    cyclic_consistency_loss_h = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX,  reduction='none')(op_h, trg_)
    cyclic_consistency_loss_nh = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX,  reduction='none')(op_nh, trg_)
    # print(cyclic_consistency_loss_h)
    # print(cyclic_consistency_loss_nh)

    l1_s = int(cyclic_consistency_loss_h.size()[0]/8)
    l2_s = int(cyclic_consistency_loss_nh.size()[0]/8)

    cyclic_consistency_loss_h = cyclic_consistency_loss_h.reshape((8,l1_s))
    cyclic_consistency_loss_nh = cyclic_consistency_loss_nh.reshape((8,l2_s))
    l1 = lab_.repeat_interleave(l1_s).reshape(8,-1)
    l2 = lab_.repeat_interleave(l2_s).reshape(8,-1)

   
    loss_ce1 = l1*cyclic_consistency_loss_h
    loss_ce2 = (l2^1)*cyclic_consistency_loss_nh

    cc_loss = (loss_ce1.mean()+loss_ce2.mean())/2
    CCC.append(cc_loss.item())

    print('AVG Reward {} || Avg Cyclic Loss {}'.format(np.mean(np.array(REWARDS)),np.mean(np.array(CCC))))

    tot_loss = rl_loss+cc_loss
    return tot_loss
















      
    


def train(model, iterator, optimizer, criterion, clip, epoch_nos):
    
    model.train()
    print('here')
    epoch_loss = 0
    
    for i_, batch in tqdm(enumerate(iterator)):
        
        src = batch.s
        trg = batch.t
        lab = batch.l
        # print(lab)
        los,lot = [],[]
        for i in src:
            s = []
            for j in i:
                s.append(SRC.vocab.itos[j.item()])
            # print(len(s))
            s = s[0:100]
            los.append(' '.join(s))

      
        output_h, _,output_nh,_, style_h,style_nh, content, pred_style,mi = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim_h = output_h.shape[-1]
            
        output_h = output_h.contiguous().view(-1, output_dim_h)

        output_dim_nh = output_nh.shape[-1]
            
        output_nh = output_nh.contiguous().view(-1, output_dim_nh)

        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        result = torch.einsum('abc, abc->ab', style_h, style_nh)
        bs = style_h.size()[0]
        ll = style_h.size()[1]
        loss_mse = nn.MSELoss()(result, torch.zeros((bs,ll)).to(device))
        print(output_h.shape,trg.shape)
       
        loss_ce1 = criterion(output_h, trg)
        loss_ce2 = criterion(output_nh, trg)
        loss_st = criterion_style(pred_style,lab)
        l1_s = int(loss_ce1.size()[0]/8)
        l2_s = int(loss_ce2.size()[0]/8)

        loss_ce1 = loss_ce1.reshape((8,l1_s))
        loss_ce2 = loss_ce2.reshape((8,l2_s))
        l1 = lab.repeat_interleave(l1_s).reshape(8,-1)
        l2 = lab.repeat_interleave(l2_s).reshape(8,-1)
        loss_ce1 = l1*loss_ce1
        loss_ce2 = (l2^1)*loss_ce2


        # print(lab)
        # print(loss_ce1)
        # print(loss_ce1.size())
        # print(loss_ce2)
        # print(loss_ce2.size())
        
        loss_ce1 = loss_ce1.mean()
        loss_ce2 = loss_ce2.mean()

        # kk = list(np.linspace(0.0, 1.0, num=len(train_iterator)))

        
        # if i_==len(iterator)-80:
        #     neg_humor_vec,pos_humor_vec = [],[]
        #     print('constructing style vectors...')
        #     for i in range(1000):
        #         style,lab = get_vectors(i,SRC,TRG,model,'cuda')
        #         if lab==0:
        #             neg_humor_vec.append(style)
        #         else:
        #             pos_humor_vec.append(style)


        #     print(len(neg_humor_vec))
        #     print(len(pos_humor_vec))
        #     print(neg_humor_vec[0])
        #     print(neg_humor_vec[0].size())


        #     neg_humor_vec = torch.stack(neg_humor_vec).squeeze(dim=1).mean(dim=0)
        #     pos_humor_vec = torch.stack(pos_humor_vec).squeeze(dim=1).mean(dim=0)

        #     print(neg_humor_vec.size())
        #     print(pos_humor_vec.size())
        #     torch.save(pos_humor_vec,'hum_vec.pt')


        if epoch_nos>=4:
            rl_loss = reinforce(batch.s,batch.t,SRC,TRG,model,device,batch.l)
            loss = 0.5*(loss_ce1+loss_ce2+loss_mse+0.0*loss_st)+0.5*rl_loss
            if i_%50==0:
                print(loss_ce1,loss_ce2,loss_mse,loss_st,mi,rl_loss)
            

            
        else:
            loss = loss_ce1+loss_ce2+loss_mse+0.0*loss_st
            if i_%50==0:
                print(loss_ce1,loss_ce2,loss_mse,loss_st,mi)
        
        
        
        
        
        # loss.backward()

        # # print(model.)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # optimizer.step()
        model.backward(loss)
        model.step()
        
        epoch_loss += (loss_ce1+loss_ce2).item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.s
            trg = batch.t
            lab = batch.l

            output_h, _, output_nh,_, style_h,style_nh, content, pred_style,_ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim_h = output_h.shape[-1]
            
            output_h = output_h.contiguous().view(-1, output_dim_h)

            output_dim_nh = output_nh.shape[-1]
            
            output_nh = output_nh.contiguous().view(-1, output_dim_nh)

            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            result = torch.einsum('abc, abc->ab', style_h, style_nh)
            bs = style_h.size()[0]
            ll = style_h.size()[1]
            loss_mse = nn.MSELoss()(result, torch.zeros((bs,ll)).to(device))
            loss_ce1 = criterion(output_h, trg).mean()
            loss_ce2 = criterion(output_nh, trg).mean()
            # loss_st = criterion_style(pred_style,lab)
            
            

            epoch_loss += (loss_ce1+loss_ce2).item()
        
    return epoch_loss / len(iterator)



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to('cuda')
    
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, evaluate=False):
    if evaluate:
        d =  OneHotCategorical(logits=logits.view(-1, latent_dim, categorical_dim))
        return d.sample().view(-1, latent_dim * categorical_dim)

    y = gumbel_softmax_sample(logits, temperature)
    return y.view(-1, latent_dim * categorical_dim)

def translate_sentence_for_baseline(sentence, src_field, trg_field, model, device,lab, max_len = 50):
    
    # model.train()
        
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:98]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    remaining = 100-len(tokens)

    tokens = tokens+[src_field.pad_token]*remaining

    # print(tokens)
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)
    
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    pred_toks = []
    W = []
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        # with torch.no_grad():
        if lab==0:
            output, attention, sample, w_grad = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, src_mask,gumbel_softmax=True)
        if lab==1:
            output, attention, sample, w_grad = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, src_mask,gumbel_softmax=True)
        
        # gumbel_one_hot = F.gumbel_softmax(output[:,-1,:].squeeze(),tau=1,hard=True)
        # ll = int(list(gumbel_one_hot.size())[0])
        # aranged = torch.arange(ll).to('cuda')
        # hard_sample_w_grad = torch.sum(aranged*gumbel_one_hot) # reparameterization to get one hot vector
        # # print(hard_sample_w_grad)
        # # token_probs = F.softmax(output[:,-1,:],dim=1) # softmax prob distribution
        # # m = Categorical(token_probs)
        # # action = m.sample()
        # # print(action)
        w_grad = w_grad.unsqueeze(0)
        pred_token = output.argmax(2)[:,-1].item()
        # print(sample)
        # print(hard_sample_w_grad,pred_token)
        # pred_toks.append(hard_sample_w_grad.unsqueeze(0))
        pred_toks.append(sample)
        W.append(w_grad)
        # print(sample,w_grad)
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    pred_toks = torch.cat(pred_toks).unsqueeze(0)

    W = torch.cat(W).unsqueeze(0)
    # print(pred_toks)
    # pred_toks = pred_toks.type(torch.LongTensor)
    # print(pred_toks)
    return trg_tokens[1:], attention, pred_toks, W

#https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # print(logits.dim())
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Here is how to use this function for top-p sampling
temperature = 1.5
top_k = 0
top_p = 0.9

def translate_sentence(sentence, src_field, trg_field, model, device,lab, max_len = 50):
    
    model.eval()
    # print('*********')
    # print(sentence)
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:18]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    remaining = 20-len(tokens)

    tokens = tokens+[src_field.pad_token]*remaining

    # print(tokens)
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)
    
    src_mask = model.make_src_mask(src_tensor)
    
    # with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)
    style_h = model.style_proj_h(enc_src)
    style_nh = model.style_proj_nh(enc_src)

    pred_style = model.style_labeler(torch.mean(style_h,dim=1))

    content = model.content_proj(enc_src)

    #style and content's resultant should reconstruct input sentence,so,
    # enc_out = style+content

   
    enc_out_h = style_h+content
    enc_out_nh = style_nh+content

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    pred_toks = []
    W = []
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
          if lab==0:
            output, attention = model.decoder_humor(trg_tensor, enc_out_h, trg_mask, src_mask)
          if lab==1:
            output, attention = model.decoder_nonhumor(trg_tensor, enc_out_nh, trg_mask, src_mask)
        
        
       
        pred_token = output.argmax(2)[:,-1].item() # uncomment this line for greedy sampling

        # logits = output[:, -1, :].squeeze() / temperature
        
        # filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        # # Sample from the filtered distribution
        # probabilities = F.softmax(filtered_logits, dim=-1)
        # pred_token = torch.multinomial(probabilities, 1).item()
 
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
   
    # print(pred_toks)
    return trg_indexes[1:], attention



from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os
device = "cuda"
# model_id = "gpt2-medium"
model_id='./gpt2_lm/checkpoint-9600'
gpt2 = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer_gpt2 = GPT2TokenizerFast.from_pretrained("gpt2-medium")

os.environ["WANDB_DISABLED"] = "true"





SRC_,TRG_,style_labs,labs = [[]],[],[],[]
REF = [[]]
PPLS = []

def get_ppl(a):
  print(a)
  p = []
  for i in a.split(' '):
    if i== '<eos>':
      break
    else:
      p.append(i)

  sent = ' '.join(p)
  print(sent)
  encodings = tokenizer_gpt2(sent, return_tensors="pt")

  max_length = gpt2.config.n_positions
  stride = 512

  nlls = []
  for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
      begin_loc = max(i + stride - max_length, 0)
      end_loc = min(i + stride, encodings.input_ids.size(1))
      trg_len = end_loc - i  # may be different from stride on last loop
      input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = gpt2(input_ids, labels=target_ids)
          neg_log_likelihood = outputs[0] * trg_len

      nlls.append(neg_log_likelihood)
  ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
  return ppl,sent


from mosestokenizer import *
TURN = 0


with open(dir_path+'./reference.0', 'r') as f:
  k_0 = f.readlines()
k_0 = list(map(lambda x: x.strip(), k_0))

with open(dir_path+'./reference.1', 'r') as f:
  k_1 = f.readlines()
k_1 = list(map(lambda x: x.strip(), k_1))

test_ref = k_0+k_1
import re
onlyref = {}




sss = []
for i, batch in enumerate(test_iterator):
  list_of_src_tokens = batch.s
  n_training_episodes = int(batch.s.size()[0])
  for i_episode in tqdm(range(n_training_episodes)):
    s = []
    for i in list_of_src_tokens[i_episode,:]:
        
        
        if SRC.vocab.itos[i.item()]=='<eos>':
            # print('reached')
            break
        s.append(SRC.vocab.itos[i.item()])

        # print(len(s))
        
    sentence = s[1:]
    sentence = ' '.join(sentence)
    sss.append(sentence)

print(len(sss))
print(len(test_src))




# test_src = test_src[0:960]
# test_ref = test_ref[0:960]
for cnt,i in enumerate(test_ref):
  onlyref[test_src[cnt]] = re.sub(re.escape(test_src[cnt]),"",i).strip()

print(onlyref)
# exit(0)

SENT = []
LABS = []

def eval(model, iterator):
    TRN = 0
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            print(batch)

            # src = batch.s
            # trg = batch.t
            # lab = batch.l
            list_of_src_tokens = batch.s
            list_of_trg_tokens = batch.t
            src_field = SRC
            trg_field = TRG
            model = model
            device = 'cuda'
            lab_ = batch.l
            max_len = 20
            n_training_episodes = int(batch.s.size()[0])
            print('n training episode {}'.format(n_training_episodes))
            # reinforce(batch.s,batch.t,SRC,TRG,model,device,batch.l)



            for i_episode in tqdm(range(n_training_episodes)):
              
              lab = lab_[i_episode].item()
              saved_log_probs = []
              rewards = []
              # sentence = list_of_src_sentences[i_episode]
              # print(sentence)
              # trg_sentence = list_of_trg_sentences[i_episode]
              # if isinstance(sentence, str):
              #     # nlp = spacy.load('de_core_news_sm')
              #     nlp = en_core_web_sm.load()
              #     tokens = [token.text.lower() for token in nlp(sentence)]
              # else:
              #     tokens = [token.lower() for token in sentence]

              s = []
              for i in list_of_trg_tokens[i_episode,:]:
                  
                  
                  if TRG.vocab.itos[i.item()]=='<pad>':
                      # print('reached')
                      break
                  s.append(TRG.vocab.itos[i.item()])

                  # print(len(s))
                  
              trg_sentence = ' '.join(s[1:-1])
              # print(trg_sentence)

              s = []
              for i in list_of_src_tokens[i_episode,:]:
                  
                  
                  if SRC.vocab.itos[i.item()]=='<eos>':
                      # print('reached')
                      break
                  s.append(SRC.vocab.itos[i.item()])

                  # print(len(s))
                  
              sentence = s[1:]

              with MosesDetokenizer('en') as detokenize:
                SRC_[0].append(detokenize(sentence))


              
              

              sentence = ' '.join(sentence)
              with MosesDetokenizer('en') as detokenize:
                tokenize = MosesTokenizer('en')
                try:
                  ref = tokenize(onlyref[sentence].lower())
                except:
                  TRN+=1
                  # ref = tokenize(onlyref["ever since joes has changed hands it 's just gotten worse and worse ."].lower())
                  ref = s[1:]

                tokenize.close()
                print(ref)
                REF[0].append(detokenize(ref))
              print('***********')
              print(sentence)
              print(REF)
              baseline,_ = translate_sentence(sentence, SRC, TRG, model, device,lab, max_len = 20)
              # print(baseline)
              baseline = [TRG.vocab.itos[i] for i in baseline]
              # print('bs_toks {}'.format(bs_toks))
              print(' '.join(baseline[:-1]), lab)
              ppl,sent = get_ppl(' '.join(baseline[:-1])) 
              print(ppl)
              SENT.append(sent)
              PPLS.append(np.nan_to_num(ppl))
              print('*********')
              # SRC_[0].append(sentence)
              with MosesDetokenizer('en') as detokenize:
                TRG_.append(detokenize(baseline[:-1]))

              # TRG_.append(' '.join(baseline[:-1]))
              LABS.append(lab)
              if lab==0:
                labs.append(1)
              else:
                labs.append(0)
              inputs = tokenizer(' '.join(baseline[:-1]), return_tensors="pt")

              with torch.no_grad():
                  logits = distilbert(**inputs, return_dict=True).logits

              pred_style = logits.argmax().item()
              style_labs.append(pred_style)

    return  TRN
              









def get_vectors(i, src_field, trg_field, model, device, max_len = 50):


    sentence = vars(train_data.examples[i])['s']
    lab = vars(train_data.examples[i])['l']
    model.eval()
            
    if isinstance(sentence, str):
        # nlp = spacy.load('de_core_news_sm')
        nlp = en_core_web_sm.load()
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = tokens[0:98]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]


    remaining = 100-len(tokens)
    tokens = tokens+[src_field.pad_token]*remaining
    
    # print(tokens)
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # print(src_indexes)

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # print(src_tensor)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
        style = model.style_proj(enc_src)

        

    return style,lab

N_EPOCHS = 10
CLIP = 1
from sklearn.metrics import f1_score,accuracy_score
best_valid_loss = float('inf')
from collections import OrderedDict
import sacrebleu


# for epoch in range(N_EPOCHS):
    
    # start_time = time.time()
    # optimizer=None
    # train_loss = train(model, train_iterator, optimizer, criterion, CLIP,epoch)
    # valid_loss = evaluate(model, test_iterator, criterion)
    
    # end_time = time.time()
    
    # epoch_mins, epoch_secs = epoch_time(start_time, end_time)

# state_dict = torch.load('./double_decoder_rl_cc_yelp_remaining/tr_model_full_reward_epoch0.pt')
state_dict = torch.load('./double_decoder_rl_cc_yelp_nll/tr_model_full_reward_epoch0.pt')
# print(state_dict)
state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
# print(state_dict)
model.load_state_dict(state_dict)
print(model)
model.to('cuda')
TRN = eval(model,test_iterator)
print(SRC_)
print(TRG_)
print(labs)
print(style_labs)
print(PPLS)
print(REF)
print(f1_score(labs,style_labs,average='macro'))
print(f1_score(labs,style_labs,average='weighted'))
print(accuracy_score(labs,style_labs))
print(sacrebleu.corpus_bleu(TRG_, SRC_))
print(sacrebleu.corpus_bleu(TRG_, REF))
print(np.mean(np.array(PPLS)))
print(TRN)
import pandas as pd

details = pd.DataFrame({
    'target': list(TRG_),
    'source': [i for i in SRC_[0]],
    'reference': [i for i in REF[0]],
    'sent_for_ppl': list(SENT),
    'label':list(LABS)
})

details.to_csv('./results_yelp_nll_eval.csv')
























