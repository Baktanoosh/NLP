# models.py

import numpy as np
import collections
import time
import math
import torch
import random
import torch.nn as nn
from torch import optim



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
        self.lstm = torch.nn.LSTM(self.dict_size, self.hidden_size, num_layers=1, dropout=0)
        self.linear = torch.nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        self.softmax = nn.LogSoftmax(dim=0)

    def set_parameters(self, embedding_size, num_layers, vocab_index, cons_size):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_index =  vocab_index
        self.cons_size = cons_size
        self.embedding = torch.nn.Embedding(self.dict_size, embedding_size)

    def forward(self, input):
        embedded_input = self.char_embedding(input)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.lstm(embedded_input, init_state)
        return self.softmax(hidden_state)
    
    def predict(self, context):
        contex_ind = [self.vocab_index.index_of(char) for char in context]
        esx = torch.from_numpy(np.array(contex_ind)).long()
        prob = self.forward(esx)
        return torch.argmax(prob).item()
            
    def char_embedding(self,input):
        embeding_indice = list()
        embeding_indice = [self.vocab_index.index_of(char) for char in input]
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
    num_epochs = 2
    num_classes = 2
    num_layers = 1
    embedding_size = 27
    dict_size = len(vocab_index)
    cons_size = len(train_cons_exs)
    vowel_size = len(train_vowel_exs)
    hidden_size = 128
    initial_learning_rate = 0.001
    contex = ''
    train_data = []
    contex = train_cons_exs + train_vowel_exs
    rnn_model = RNNClassifier(dict_size, hidden_size)
    rnn_model.set_parameters(embedding_size, num_layers, vocab_index, cons_size)
 
    for i, train_ex in enumerate(contex):
        if i < cons_size:
            label = 0
        else:
            label = 1
        label = torch.FloatTensor([[label]]) 
        train_data.append((train_ex, label))
    optimizer = optim.Adam(rnn_model.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        contex_ind = [vocab_index.index_of(char) for char in contex]
        esx = torch.from_numpy(np.array(contex_ind)).long()
        log_probs = rnn_model.forward(esx)
        loss = torch.neg(log_probs).dot(label)
        total_loss += loss
        loss.backward()
        optimizer.step()
    now = time.time()
    print("Total Traing time", now-start)
    return rnn_model

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


class RNNLanguageModel(LanguageModel):
    def __init__(self):
        raise Exception("Implement me")

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")
