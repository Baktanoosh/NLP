"""
from sentiment_data import *
from utils import *
from collections import Counter
from typing import List

# & C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/optimization.py --lr 0.1
# & C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/neural_sentiment_classifier.py --model TRIVAL --no_run_on_test
# C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/neural_sentiment_classifier.py --word_vecs_path data/glove.6B.300d-relativized.txt

import torch
x = torch.rand(5, 3)
print(x)
"""
import torch
import torch.nn as nn
import numpy as np

from models import SentimentClassifier, TrivialSentimentClassifier, NeuralSentimentClassifier


num_epochs = 10
num_classes = 2
feat_vec_size = 300
initial_learning_rate = 0.001
embedding_size = 6
sentence = ['kir','khar','to','koonet']
train_xs = np.array([0, 0, 0, 1], dtype=np.float32)

ex = torch.from_numpy(sentence).float()

nsc = NeuralSentimentClassifier(feat_vec_size, embedding_size, num_classes)
print(nsc.predict(ex))

