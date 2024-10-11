# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import math


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, d_hidden, d_ff, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.d_model = d_model          # embedding dimension
        self.d_internal = d_internal
        self.d_hidden = d_hidden
        self.d_ff = d_ff
        self.num_positions = num_positions
        self.num_classes = num_classes
        self.g = nn.ReLU()          # nonlinear function g(x)
        self.W1 = nn.Linear(d_model, d_ff)        # weight vector W1 for FFN
        self.W2 = nn.Linear(d_ff, num_classes)        # weight vector W2 for FFN
        self.log_softmax = nn.LogSoftmax(dim = -1)   
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        #nn.init.xavier_uniform_(self.W1.weight)
        #nn.init.xavier_uniform_(self.W2.weight)
        self.W3 = nn.Linear(d_model, num_classes)  
        nn.init.xavier_uniform_(self.W3.weight)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x = self.word_embedding(indices)        # torch.Size([num_positions, d_model])
        pos_enc = PositionalEncoding(self.d_model, self.num_positions, batched=False)
        x = x + pos_enc.forward(x)
        Transformer_layer =  nn.ModuleList([TransformerLayer(self.d_model, self.d_internal, self.d_hidden, self.d_ff)])
        (output, attention_score) =  nn.Sequential(*Transformer_layer).forward(x)      # output = sen_len x d_model; self-attention scores: sen_len x d_v
        #Transformer_layer2 =  TransformerLayer(self.d_model, self.d_internal, self.d_hidden, self.d_ff) 
        #(output, attention_score) =  Transformer_layer2.forward(output)      # output = sen_len x d_model; self-attention scores: sen_len x d_v
        #output = self.log_softmax(self.W2(self.g(self.W1(output))))          # FFN NLLLoss
        #output = self.log_softmax(self.W3(output))          # FFN
        #output = self.W2(self.g(self.W1(output)))          # FFN CrossEntropy
        #print("done", attention_score.shape, output.shape)      # output = sen_len x d_model; self-attention scores: sen_len x num_classes
        return (output, [attention_score])

# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, d_hidden, d_ff):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # self.d_model = d_model          # embedding dimension
        self.d_internal = d_internal
        self.w_query = nn.Linear(d_model, d_internal)        # d_model x d_internal or d_k
        self.w_key = nn.Linear(d_model, d_internal)        # d_model x d_internal or d_k
        self.w_value = nn.Linear(d_model, d_model)        # d_model x d_hidden (d_v)
        self.g = nn.ReLU()          # nonlinear function g(x)
        self.W1 = nn.Linear(d_model, d_ff)        # weight vector W1 for FFN, d_hidden (d_v) x  d_ff
        self.W2 = nn.Linear(d_ff, d_model)        # weight vector W2 for FFN, d_ff x d_model
        self.softmax = nn.Softmax(dim = -1)   
        #nn.init.xavier_uniform_(self.w_query.weight)
        #nn.init.xavier_uniform_(self.w_key.weight)
        #nn.init.xavier_uniform_(self.w_value.weight)
        #nn.init.xavier_uniform_(self.W2.weight)
        #nn.init.xavier_uniform_(self.W1.weight)

    def forward(self, x):
        Q = self.w_query(x)
        K = self.w_key(x)
        V = self.w_value(x)
        attention_score = torch.matmul(Q, K.transpose(1,0))/math.sqrt(self.d_internal)      # sen_len x sen_len
        attention_score = self.softmax(attention_score)        #  self-attention scores
        map = torch.matmul(attention_score, V)   # sen_len x d_model or d_v
        map = map + x         # adding residual
        output = self.W2(self.g(self.W1(map)))          # FFN: sen_len x d_model 
        #print("done", map.shape, attention_score.shape, output.shape)
        output = output + map         # adding residual
        return (output, attention_score)

# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    begin = time.time()
    print("________________________________________________________________________________")
    # hyperparameters   
    vocab_size = 27         # vocabulary size of the embedding layer
    num_positions = 20      # max sequence length that will be fed to the model
    d_model = 80            # dimension of the inputs and outputs of the transformer layer
    d_internal = 20          # size of the Query & Key
    d_hidden =  d_model          # size of the Value
    d_ff = 150               # size of the hidden layer in FFN
    num_classes = 3          # number of classes/output predicted at the output layer
    num_layers = 2          # number of TransformerLayers to use
    lr = 1e-4               # learning rate
    num_epochs = 5
    batch_size = 1
    dropout = 0.        # only if there are more than 1 layer
    # training data & model setup
    ex_idxs = [i for i in range(0, len(train))]
    model = Transformer(vocab_size, num_positions, d_model, d_internal, d_hidden, d_ff, num_classes, num_layers)
    model.train()
    loss_fcn = nn.NLLLoss() 
    #loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        random.shuffle(ex_idxs)
        for ex_idx in ex_idxs:
            #loss = 0.
            x = train[ex_idx].input_tensor
            target = train[ex_idx].output_tensor
            model.zero_grad()
            (log_probs, attn_vals) = model(x)         # log_probs: torch.Size([20, 3]), attn_vals: torch.Size([20, 20])
            #print(log_probs)
            #exit()
            loss = loss_fcn(log_probs, target)
            loss_this_epoch += loss.item()
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (t, loss_this_epoch))
        decode(model, train[0:100])
    model.eval()
    end = time.time()
    print(f"Total runtime of the program is {end - begin}")
    #print("map: ", attn_vals.shape)
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        #"""
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                #plt.savefig("plots/%i_attns%i.png" % (i, j))
                plt.savefig("C:/Users/axb5786/Desktop/a5-distrib/plots/%i_attns%i.png" % (i, j))
        #"""        
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
