import csv
import os

import re
import unicodedata
import string
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
import torch.nn.utils as utils

import numpy as np

# Pre-processing

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

class Preprocessing:
    def __init__(self):
        self.word_count = {}
        self.vocab_size = 3
        self.word_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.index_to_word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

    def process_fact(self, s):
        s = s.lower().strip()
        words = s.split(' ')
        for w in words:
            if w not in self.word_to_idx:
                self.word_to_idx[w] = self.vocab_size
                self.index_to_word[self.vocab_size] = w
                self.vocab_size += 1
                self.word_count[w] = 1
            else:
                self.word_count[w] += 1

with open('Data/data10K.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=';')
    preprocess = Preprocessing()
    for line in csvreader:
        facts = line[2:len(line)]
        for fact in facts:
            preprocess.process_fact(fact)
print("Vocab size = {}".format(preprocess.vocab_size))

# Define dimensions
INPUT_DIM =  preprocess.vocab_size
OUTPUT_DIM =  preprocess.vocab_size  # This is the output vocabulary size
EMBEDDING_DIM = 100
HIDDEN_DIM = 100

MAX_CONTEXT_LEN = 60
MAX_FACT_LEN = 60
MAX_DESC_LEN = 60
BATCH_SIZE = 100

USE_CUDA = False
if cuda.is_available():
    USE_CUDA = True

#****************************** INPUT MODULE ********************************************
#****************************************************************************************
class PositionalFactEncoder(nn.Module):
    def __init__(self):
        super(PositionalFactEncoder, self).__init__()

    def forward(self, embedded_sentence, fact_lengths):

        _, slen, elen = embedded_sentence.size()

        l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
        l = torch.FloatTensor(l)
        l = l.unsqueeze(0)
        l = l.expand_as(embedded_sentence)
        l = Variable(l)
        if USE_CUDA: l = l.cuda()
        weighted = embedded_sentence * l
        return torch.sum(weighted, dim=1).squeeze(1)


class FactEncoder(nn.Module):
    def __init__(self):
        super(FactEncoder, self).__init__()
        self.gru = nn.GRU(input_size = EMBEDDING_DIM, hidden_size = HIDDEN_DIM, dropout = 0.1, batch_first = True)
        self.hidden = self.init_hidden()

    def forward(self, embedded_sentence, fact_lengths):
        gru_out, self.hidden = self.gru(embedded_sentence, self.hidden)
        fact_embeddings = Variable(torch.FloatTensor(np.zeros([MAX_CONTEXT_LEN, EMBEDDING_DIM])))
        if USE_CUDA: fact_embeddings = fact_embeddings.cuda()
        for i in range(len(fact_lengths)):
            fact_embeddings[i] = gru_out[i,fact_lengths[i]-1,:].view(1, -1)
        return fact_embeddings

    def init_hidden(self):
        h_0 = Variable(torch.zeros(1, MAX_CONTEXT_LEN, HIDDEN_DIM))
        if USE_CUDA: h_0 = h_0.cuda()
        return h_0

class InputModule(nn.Module):
    def __init__(self):
        super(InputModule, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings = INPUT_DIM,embedding_dim = EMBEDDING_DIM, padding_idx=0)
        self.fact_encoder = PositionalFactEncoder()

    def forward(self, context, fact_lengths):
        context_len, max_fact_len = context.size()
        embedded_context = self.word_embeddings(context)
        embedded_context = embedded_context.view(context_len, max_fact_len, -1)
        encoded_facts = self.fact_encoder(embedded_context, fact_lengths)
        return encoded_facts

#*********************************** OUTPUT MODULE ************************************************
#**************************************************************************************************

class OutputModule(nn.Module):
    def __init__(self):
        super(OutputModule, self).__init__()
        self.gru = nn.GRU(input_size = EMBEDDING_DIM , hidden_size = HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, decoder_input, hidden):
        decoder_input = decoder_input.unsqueeze(1)
        gru_out, hidden = self.gru(decoder_input, hidden)
        output = F.log_softmax(self.output(gru_out.squeeze(1)))
        return output, hidden

def init_hidden():
    h_0 = Variable(torch.zeros(1, 1, HIDDEN_DIM))
    if USE_CUDA: h_0 = h_0.cuda()
    return h_0

class Seq2seq(nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.word_embeddings = nn.Embedding(num_embeddings = INPUT_DIM,embedding_dim = EMBEDDING_DIM, padding_idx=0)
        self.input_module = InputModule()
        self.output_module = OutputModule()
        self.linear_1 = nn.Linear(MAX_DESC_LEN*HIDDEN_DIM, HIDDEN_DIM)

    def forward(self,context, fact_lengths, description, flag):
        fact_embeddings = self.input_module(context, fact_lengths)

        # Initialize the hidden state of the output sequence
        output_hidden = self.linear_1(fact_embeddings.view(1, -1)).unsqueeze(0)
        desc_len = description.size()[0]
        if flag == 'training':
            # Obtain the embedding of the input word
            word_input = Variable(torch.LongTensor([[1]]))
            if USE_CUDA: word_input = word_input.cuda()
            embedding = self.word_embeddings(word_input).squeeze(1)

            loss = Variable(torch.FloatTensor([0.0]))
            if USE_CUDA: loss = loss.cuda()
            #******************** Unfold the output sequence *************
            for idx in range(desc_len):
                decoder_input = embedding
                output, output_hidden = self.output_module(decoder_input, output_hidden)

                #***************** Calculate Loss ***********************
                y_true = description[idx]
                y_pred = output
                loss += loss_function(y_pred, y_true)

                #***************** Prepare Next Decoder Input **************************
                word_input = description[idx].view(1,-1)
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input).squeeze(1)
            return loss

        if flag == 'test' or flag == 'validation':
            word_input = Variable(torch.LongTensor([[1]]))
            if USE_CUDA: word_input = word_input.cuda()
            embedding = self.word_embeddings(word_input).squeeze(1)

            decoded_words = []
            for idx in range(MAX_DESC_LEN):
                decoder_input = embedding
                output, output_hidden = self.output_module(decoder_input, output_hidden)

                # Interpret the decoder output
                value, index =  output.data.topk(1)
                index = index[0][0]
                #***************** Prepare Next Decoder Input **************************
                word_input = Variable(torch.LongTensor([[index]]))
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input).squeeze(1)

                if index == preprocess.word_to_idx['<EOS>']:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(preprocess.index_to_word[index])

            return decoded_words

# ****************** DEFINE MODEL, LOSS FUNCTION and OPTIMIZATION PROCESS ***********************
loss_function = nn.NLLLoss()
model = Seq2seq()
if USE_CUDA: model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#*************************** Training **********************************

def prepare_input_seq(f):
    f = f.lower().strip()
    f = f + ' <EOS>' #Append the special symbol to indicate end of source  sentence

    words = f.split(' ')
    idx_seq = []
    for w in words:
        if w in preprocess.word_to_idx:
            idx_seq.append(preprocess.word_to_idx[w])
    return idx_seq, len(idx_seq)

contexts = []
descriptions = []
flag = 'training'
for epoch in range(25):
    with open('Data/training.csv','r') as f:
        csvreader = csv.reader(f, delimiter=';')
        row_count = 0
        training_loss = 0.0
        for row in csvreader:

            # ******************  CONSTRUCT TENSORS FOR INPUT CONTEXT **********
            facts = row[2:len(row)-1]
            if len(facts) > MAX_CONTEXT_LEN:
                facts = row[2: MAX_CONTEXT_LEN]
            context = np.zeros((MAX_CONTEXT_LEN, MAX_FACT_LEN), dtype=np.int)
            fact_lengths = []
            for i, fact in enumerate(facts):
                input_seq, input_seq_len = prepare_input_seq(fact)
                input_seq = np.array(input_seq)
                context[i] = np.pad(input_seq, (0, MAX_FACT_LEN - len(input_seq)), 'constant', constant_values = 0)
                fact_lengths.append(input_seq_len)

            context = np.asarray(context, dtype=np.int)
            context = Variable(torch.LongTensor(context))
            if USE_CUDA: context = context.cuda()

            # *************  CONSTRUCT TENSORS FOR OUTPUT DESCRIPTION **********
            desc = row[len(row)-1]
            desc, desc_len = prepare_input_seq(desc)
            desc = np.array(desc)
            description = Variable(torch.LongTensor(desc))
            if USE_CUDA: description = description.cuda()

            # ***********  UPDATE THE MODEL WITH THE MINI-BATCH of SIZE 1 ******
            optimizer.zero_grad()
            loss = model(context, fact_lengths, description, flag)
            loss.backward()
            utils.clip_grad_norm(model.parameters(), 40)
            training_loss += loss.data[0]
            optimizer.step()

            row_count += 1
            #if row_count == 8:  # Training : Validation : Test 8:1:1
            #    break
        print("Epoch {}, Loss {}".format(epoch+1, training_loss/float(row_count)))
    f.close()

#*****************************  Test ******************************************************
flag = 'test'
print("Testing...")
fout = open('result_Fact2seq.csv', 'w+')
with open('Data/test.csv','r') as f:
    csvreader = csv.reader(f, delimiter=';')
    row_count = 0
    validation_loss = 0.0
    for row in csvreader:
        # ******************  CONSTRUCT TENSORS FOR INPUT CONTEXT ***********************
        facts = row[2:len(row)-1]
        if len(facts) > MAX_CONTEXT_LEN:
            facts = row[2: MAX_CONTEXT_LEN]
        context = np.zeros((MAX_CONTEXT_LEN, MAX_FACT_LEN), dtype=np.int)
        fact_lengths = []
        for i, fact in enumerate(facts):
            input_seq, input_seq_len = prepare_input_seq(fact)
            input_seq = np.array(input_seq)
            context[i] = np.pad(input_seq, (0, MAX_FACT_LEN - len(input_seq)), 'constant', constant_values = 0)
            fact_lengths.append(input_seq_len)

        context = np.asarray(context, dtype=np.int)
        context = Variable(torch.LongTensor(context))
        if USE_CUDA: context = context.cuda()

        # ******************* JUST A DUMMY DESCRIPTION ************************
        description = Variable(torch.LongTensor([0]))
        if USE_CUDA: description = description.cuda()

        generated_desc = model(context, fact_lengths, description, flag)
        generated_desc = ' '.join(generated_desc[0:len(generated_desc) - 1])
        fout.write(row[len(row)-1]+ ';' + generated_desc + '\n')

        row_count += 1
        #if row_count == 2:
        #    break
f.close()
fout.close()
