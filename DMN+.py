import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.cuda as cuda
import numpy as np
import torch.nn.utils as utils

import csv

#*****************************************************************************************
#  This implementation is adapted from the DMN+ implementation of Wonjae Kim
#  https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch/blob/master/babi_main.py
#******************************************************************************************

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

with open('Data/data10KDMN+.csv', 'r') as f:
    csvreader = csv.reader(f, delimiter=';')
    preprocess = Preprocessing()
    for line in csvreader:
        facts = line[1:len(line)]
        for fact in facts:
            preprocess.process_fact(fact)
print("Vocab size = {}".format(preprocess.vocab_size))

# Define dimensions
vocab_size = preprocess.vocab_size
hidden_size = 80
embedding_dim = 80

MAX_CONTEXT_LEN = 60
MAX_FACT_LEN = 60
MAX_DESC_LEN = 60


USE_CUDA = False
if cuda.is_available():
    USE_CUDA = True

def position_encoding(embedded_sentence):
    _, _, slen, elen = embedded_sentence.size()

    l = [[(1 - s/(slen-1)) - (e/(elen-1)) * (1 - 2*s/(slen-1)) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l)
    l = l.unsqueeze(0) # for #batch
    l = l.unsqueeze(1) # for #sen
    l = l.expand_as(embedded_sentence)
    if USE_CUDA: l = l.cuda()
    weighted = embedded_sentence * Variable(l)
    return torch.sum(weighted, dim=2).squeeze(2) # sum with token

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size))
        if USE_CUDA: C = C.cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C

class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)
        init.xavier_normal(self.z1.state_dict()['weight'])
        init.xavier_normal(self.z2.state_dict()['weight'])
        init.xavier_normal(self.next_mem.state_dict()['weight'])

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = F.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions

class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        for name, param in self.gru.state_dict().items():
            if 'weight' in name: init.xavier_normal(param)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = Variable(torch.zeros(2, batch_num, self.hidden_size))
        if USE_CUDA: h0 = h0.cuda()
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts

class OutputModule(nn.Module):
    def __init__(self):
        super(OutputModule, self).__init__()
        self.gru = nn.GRU(input_size = embedding_dim , hidden_size = hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input, hidden):
        decoder_input = decoder_input
        gru_out, hidden = self.gru(decoder_input, hidden)
        output = F.log_softmax(self.output(gru_out.squeeze(1)), dim=1)
        return output, hidden

class AnswerModule(nn.Module):
    def __init__(self):
        super(AnswerModule, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output_module = OutputModule()
    def forward(self, memories, answers, mode):
        if mode == 'training':
            batch_size = answers.size()[0]
            # Initialize the training loss for the batch
            loss = Variable(torch.FloatTensor([0.0]))
            if USE_CUDA: loss = loss.cuda()
            for i in range(batch_size):
                word_input = Variable(torch.LongTensor([[1]]))
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input)
                #******************** Unfold the output sequence *************
                description = answers[i]
                output_hidden = memories[i]
                output_hidden = output_hidden.unsqueeze(0)
                for idx in range(MAX_DESC_LEN):
                    decoder_input = embedding
                    output, output_hidden = self.output_module(decoder_input, output_hidden)

                    #***************** Calculate Loss ***********************
                    y_true = description[idx]
                    y_pred = output
                    loss += loss_function(y_pred, y_true)

                    #***************** Prepare Next Decoder Input **************************
                    word_input = description[idx].view(1,-1)
                    if USE_CUDA: word_input = word_input.cuda()
                    embedding = self.word_embeddings(word_input)
            return loss

        if mode == 'test' or mode == 'validation':
            batch_size = questions.size()[0]

            for i in range(batch_size):
                word_input = Variable(torch.LongTensor([[1]]))
                if USE_CUDA: word_input = word_input.cuda()
                embedding = self.word_embeddings(word_input)

                decoded_words = []

                output_hidden = memories[i]
                output_hidden = output_hidden.unsqueeze(0)
                for idx in range(MAX_DESC_LEN):
                    decoder_input = embedding
                    output, output_hidden = self.output_module(decoder_input, output_hidden)

                    # Interpret the decoder output
                    value, index =  output.data.topk(1) # returns value and index of the maximum probability in 1 X 1 tensors
                    index = index[0][0]
                    #***************** Prepare Next Decoder Input **************************
                    word_input = Variable(torch.LongTensor([[index]]))
                    if USE_CUDA: word_input = word_input.cuda()
                    embedding = self.word_embeddings(word_input)

                    if index == preprocess.word_to_idx['<EOS>']:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(preprocess.index_to_word[index])
            return decoded_words


def init_hidden():
    h_0 = Variable(torch.zeros(1, 1, HIDDEN_DIM))
    if USE_CUDA: h_0 = h_0.cuda()
    return h_0


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        init.uniform(self.word_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule()

    def forward(self, contexts, questions, answers, mode):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        if mode == 'training':
            loss = self.answer_module(M, answers, mode)
            return loss
        else:
            predicted_desc = self.answer_module(M, answers, mode)
            return predicted_desc


def prepare_input_seq(f):
    f = f.lower().strip()
    f = f + ' <EOS>' #Append the special symbol to indicate end of source  sentence

    words = f.split(' ')
    idx_seq = []
    for w in words:
        if w in preprocess.word_to_idx:
            idx_seq.append(preprocess.word_to_idx[w])
    return idx_seq

def prepare_data(batch_data):
    contexts = []
    questions = []
    answers = []
    for i,row in enumerate(batch_data):
        # ******************  CONSTRUCT TENSORS FOR INPUT CONTEXT ***********************
        facts = row[1:len(row)-2]
        if len(facts) > MAX_CONTEXT_LEN:
            facts = row[1: MAX_CONTEXT_LEN]
        context = np.zeros((MAX_CONTEXT_LEN, MAX_FACT_LEN), dtype=np.int)
        for i, fact in enumerate(facts):
            input_seq = np.array(prepare_input_seq(fact))
            context[i] = np.pad(input_seq, (0, MAX_FACT_LEN - len(input_seq)), 'constant', constant_values = 0)
        context = np.asarray(context, dtype=np.int)
        contexts.append(context)

        # *****************  CONSTRUCT TENSORS FOR OUTPUT DESCRIPTION ********************
        desc = row[len(row)-2]
        desc = np.array(prepare_input_seq(desc))
        answers.append(np.pad(desc, (0, MAX_DESC_LEN - len(desc)), 'constant', constant_values = 0))


        # *****************  CONSTRUCT TENSORS FOR INPUTQUESTION ********************
        question = row[len(row)-1]
        question = np.array(prepare_input_seq(question))
        questions.append(np.pad(question, (0, MAX_DESC_LEN - len(question)), 'constant', constant_values = 0))

    contexts = torch.LongTensor(contexts)
    answers = torch.LongTensor(answers)
    questions = torch.LongTensor(questions)
    return contexts, questions, answers

if __name__ == '__main__':
    loss_function = nn.NLLLoss()
    model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=None)
    if USE_CUDA: model = model.cuda()
    optim = torch.optim.Adam(model.parameters())
# ****************** TRAINING **************************************************
    for epoch in range(25):
        mode = 'training'
        row_count = 0
        batch_data =[]
        total_loss = 0
        with open('Data/trainingDMN+.csv','r') as f:
            csvreader = csv.reader(f, delimiter=';')
            for row in csvreader:
                batch_data.append(row)
                row_count += 1
                if row_count%1 == 0:
                    optim.zero_grad()
                    contexts, questions, answers = prepare_data(batch_data)
                    batch_size = contexts.size()[0]
                    contexts = Variable(contexts)
                    if USE_CUDA: contexts = contexts.cuda()
                    questions = Variable(questions)
                    if USE_CUDA: questions = questions.cuda()
                    answers = Variable(answers)
                    if USE_CUDA: answers = answers.cuda()

                    loss= model(contexts, questions, answers, mode)
                    loss.backward()
                    utils.clip_grad_norm(model.parameters(), 40)
                    total_loss += loss.data[0]

                    batch_data = []
                    optim.step()
                #if row_count == 5:
                #    break
            print("Epoch={}, Loss={}".format(epoch+1, total_loss/float(row_count)))
        f.close()
#************************ TESTING ********************************************
    print("Testing...")
    mode = 'test'
    fout = open('result_DMN+.csv','w+')
    row_count = 0
    batch_data =[]
    with open('Data/testDMN+.csv','r') as f:
        csvreader = csv.reader(f, delimiter=';')
        for row in csvreader:
            batch_data.append(row)

            contexts, questions, answers = prepare_data(batch_data)
            batch_size = contexts.size()[0]
            contexts = Variable(contexts)
            if USE_CUDA: contexts = contexts.cuda()
            questions = Variable(questions)
            if USE_CUDA: questions = questions.cuda()
            answers = Variable(answers)
            if USE_CUDA: answers = answers.cuda()

            generated_desc = model(contexts, questions, answers, mode)
            batch_data = []
            generated_desc = ' '.join(generated_desc[0:len(generated_desc) - 1])
            fout.write(row[len(row)-2]+ ';' + generated_desc + '\n')

            row_count += 1
            #if row_count == 2:
            #    break
    f.close()
