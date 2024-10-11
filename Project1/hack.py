"""
from sentiment_data import *
from utils import *
from collections import Counter
from typing import List

from models import UnigramFeatureExtractor
from models import BigramFeatureExtractor
from sentiment_classifier import evaluate, train_model

sentence = ['kir','khar','to','koonet']
unigram =  UnigramFeatureExtractor(Indexer())
print("Unigram")
print(unigram.extract_features(sentence))
print("lenght", unigram.get_indexer_len())

Bigram =  BigramFeatureExtractor(Indexer())
print("Bigram")
print(Bigram.extract_features(sentence))

print("Perceptron")
#C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week1/Project1/sentiment_classifier.py --model PERCEPTRON --feats UNIGRAM
#C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week1/Project1/sentiment_classifier.py --model LR --feats UNIGRAM
#C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week1/Project1/sentiment_classifier.py --model LR --feats BIGRAM
#C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week1/Project1/sentiment_classifier.py --model LR --feats BETTER

# & C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/optimization.py --lr 0.1
# & C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/neural_sentiment_classifier.py --model TRIVAL --no_run_on_test
# & C:/Users/bakta/AppData/Local/Programs/Python/Python38/python.exe c:/Users/bakta/OneDrive/Desktop/NLP/Week3/Project2/neural_sentiment_classifier.py --word vecs path data/glove.6B.300d-relativized.txt

import torch
x = torch.rand(5, 3)
print(x)
"""

from models import SentimentClassifier, TrivialSentimentClassifier, NeuralSentimentClassifier

sentence = ['kir','khar','to','koonet']

print(form_input(sentence))

