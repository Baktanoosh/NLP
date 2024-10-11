# models.py

import numpy as np
import collections
import time
import math
import torch
import torch.nn as nn


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


class RNNClassifier(ConsonantVowelClassifier):

    def __init__(self, dict_size, hidden_size, embedding_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dict_size = dict_size
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(self.dict_size, self.embedding_size)
        self.lstm = torch.nn.LSTM(self.dict_size, self.hidden_size)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_layers)
        nn.init.xavier_uniform_(self.lstm)
        nn.init.xavier_uniform_(self.fc)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input):
        embedded_input = self.embedding(input)
        embedded_input = embedded_input.unsqueeze(1)
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.fc(embedded_input, init_state)
        return output, hidden_state, cell_state
    
    def predict(self, context):
        for idx in range(0, len(context)):
            avg = self.average_embedding(context)
            log_probs = self.forward(torch.from_numpy(avg).float())
            prediction = torch.argmax(log_probs)
        return prediction.item() 

         
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
    num_epochs = 10
    num_classes = 2
    num_layers = 1
    embedding_size = 27
    dict_size = len(vocab_index)
    cons_size = len(train_cons_exs)
    vowel_size = len(train_vowel_exs)
    hidden_size = 128
    initial_learning_rate = 0.001
    contex = ''

    contex = train_cons_exs + train_vowel_exs
    for i in range(len(contex)):
        if i < cons_size:
            label = 0
        else:
            label = 1
        label = torch.FloatTensor([[label]]) 
 
    rnn_model = RNNClassifier(dict_size, hidden_size, embedding_size, num_layers)
    hidden = rnn_model.initHidden()
    optimizer = optim.Adam(rnn_model.parameters(), lr=initial_learning_rate)
    x = [torch.from_numpy(np.array(contex)).long()]

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(contex))]
        random.shuffle(contex)
        total_loss = 0.0
        for idx in ex_indices:
            y = contex[idx].label
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            log_probs = rnn_model.forward(x)
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    now = time.time()
    print("Total Traing time", now-start)
    return output, loss.item()

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
