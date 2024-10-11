# models.py

import numpy as np
import collections
import time
import math
import torch
import random
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNNClassifier(ConsonantVowelClassifier, nn.Module):

    def __init__(self, dict_size, hidden_size):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.lstm = nn.LSTM(27, self.hidden_size, num_layers=1, dropout=0)
        self.linear = nn.Linear(self.hidden_size, 2)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        self.softmax = nn.LogSoftmax(dim=0)

    def set_parameters(self, embedding_size, num_layers, vocab_index, cons_size):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_index =  vocab_index
        self.cons_size = cons_size
        self.embedding = nn.Embedding(self.dict_size, self.embedding_size)

    def forward(self, input):
        embedded_input = self.embedding(input)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.lstm(embedded_input, init_state)
        return self.softmax(self.linear(hidden_state.squeeze(0)).squeeze(0))
    
    def predict(self, context):
        contex_ind = [self.vocab_index.index_of(char) for char in context]
        esx = torch.from_numpy(np.array(contex_ind)).long()
        prob = self.forward(esx)
        return torch.argmax(prob).item()
            
    def char_embedding(self,input):
        embeding_indice = list()
        embeding_indice = [self.vocab_index.index_of(ch) for ch in input]
        return torch.LongTensor(embeding_indice) 


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    start = time.time()
    num_epochs = 25
    num_classes = 2
    num_layers = 1
    embedding_size = 27
    dict_size = len(vocab_index)
    cons_size = len(train_cons_exs)
    vowel_size = len(train_vowel_exs)
    hidden_size = 54
    initial_learning_rate = 0.001
    cons_size = len(train_cons_exs)
    contex = train_cons_exs + train_vowel_exs
    train_data = []
    model_rnn = RNNClassifier(len(vocab_index), hidden_size).train()
    model_rnn.set_parameters(embedding_size, num_layers, vocab_index, cons_size)

    for i, train_ex in enumerate(contex):
        label = 0 if i < cons_size else 1
        train_ex = model_rnn.char_embedding(train_ex)
        train_data.append((train_ex, label))

    #optimizer = torch.optim.SGD(model_rnn.parameters(), lr=initial_learning_rate, momentum=0.9)
    optimizer = optim.Adam(model_rnn.parameters(), lr= initial_learning_rate)
    for epoch in range(0, num_epochs):
        random.shuffle(train_data)
        contex_ind = [vocab_index.index_of(ch) for ch in contex]
        for ex, label in train_data:
            optimizer.zero_grad()
            ex_tensor = torch.from_numpy(np.array(ex)).long()
            log_probs = model_rnn.forward(ex_tensor)
            y = label
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            loss = torch.neg(log_probs).dot(y_onehot)
            loss.backward()
            optimizer.step()
        now = time.time()
        print("Total Traing time", now-start)
        return model_rnn



#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, dict_size, hidden_size):
        super(RNNLanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.lstm = nn.LSTM(27, self.hidden_size, num_layers=1, dropout=0)
        self.linear = nn.Linear(self.hidden_size, 27)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        self.softmax = nn.LogSoftmax(dim=0)

    def set_parameters(self, embedding_size, num_layers, vocab_index):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_index =  vocab_index
        self.embedding = nn.Embedding(self.dict_size, self.embedding_size)

    def forward(self, input):
        embedded_input = self.embedding(input)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.lstm(embedded_input, init_state)
        return self.softmax(self.linear(output[0]).squeeze(0))
    
    def char_embedding(self,input):
        embeding_indice = list()
        embeding_indice = [self.vocab_index.index_of(ch) for ch in input]
        return torch.LongTensor(embeding_indice) 

    def get_next_char_log_probs(self, context):
        contex_ind = [self.vocab_index.index_of(char) for char in context]
        esx = torch.from_numpy(np.array(contex_ind)).long()
        prob = self.forward(esx)
        return prob.detach().numpy()

        
    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        self.next_chars = next_chars
        for ind in self.next_chars:
            log_prob = self.get_next_char_log_probs(context)
            total_log_prob += log_prob[self.vocab_index.index_of(ind)]
            context += ind
        return float(total_log_prob)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    start = time.time()
    num_epochs = 25
    num_classes = 27
    num_layers = 1
    embedding_size = 27
    chunk_len = 5
    hidden_size = 54
    initial_learning_rate = 0.001
    train_data = []
    model_rnn_lan = RNNLanguageModel(len(vocab_index), hidden_size).train()
    model_rnn_lan.set_parameters(embedding_size, num_layers, vocab_index)

    for i in range(len(train_text)):
        train_ex = train_text[i:i+5]
        label = train_text[i+1:i+6]
        if len(train_ex) != chunk_len:
            break
        train_data.append((model_rnn_lan.char_embedding(train_ex), model_rnn_lan.char_embedding(train_ex)))

    optimizer = optim.Adam(model_rnn_lan.parameters(), lr= initial_learning_rate)
    for epoch in range(0, num_epochs):
        random.shuffle(train_data)
        contex_ind = [vocab_index.index_of(char) for char in train_text]
        for ex, label in train_data:
            optimizer.zero_grad()
            ex_tensor = torch.from_numpy(np.array(ex)).long()
            log_probs = model_rnn_lan.forward(ex_tensor)
            y = label
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            loss = torch.neg(log_probs).dot(y_onehot)
            loss.backward()
            optimizer.step()
        now = time.time()
        print("Total Traing time", now-start)
        return model_rnn_lan 
