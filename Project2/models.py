# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of thias assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, inp, hid, out, word_embeddings):
        super(NeuralSentimentClassifier, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.word_embeddings = word_embeddings
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        return self.log_softmax(self.W(self.g(self.V(x))))
    
    def predict(self, ex_words):
        train_correct = 0
        for idx in range(0, len(ex_words)):
            avg = self.average_embedding(ex_words)
            log_probs = self.forward(form_input(avg))
            prediction = torch.argmax(log_probs)
        return prediction.item() 

    def average_embedding(self, x):
        average_embeding = np.sum(np.array([self.word_embeddings.get_embedding(i) for i in x ]), axis=0)/ len(x)
        return average_embeding
        

def form_input(x) -> torch.Tensor:            
    return torch.from_numpy(x).float()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    num_epochs = 10
    num_classes = 2
    feat_vec_size = 300
    initial_learning_rate = 0.001
    embedding_size = 1
    train_ex = train_exs.copy()
    dev = dev_exs.copy()

    embedding_size = word_embeddings.get_embedding_length()
    nsc = NeuralSentimentClassifier(feat_vec_size, embedding_size, num_classes, word_embeddings)
    input_size = word_embeddings.get_initialized_embedding_layer()
    optimizer = optim.Adam(nsc.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            avg = nsc.average_embedding(train_exs[idx].words)
            x = form_input(avg)
            y = train_exs[idx].label
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            nsc.zero_grad()
            log_probs = nsc.forward(x)
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return nsc