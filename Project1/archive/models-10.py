# models.py

from sentiment_data import *
from utils import *
from collections import Counter
import numpy as np
from scipy import sparse
import random 
import re
import nltk
from nltk.corpus import stopwords

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def get_indexer_len(self):
        return len(self.indexer)
    
    def set_train_indexer(self, train_exs):
        for i in range(len(train_exs)):
            for j in range(len(train_exs[i].words)):
                self.indexer.add_and_get_index(train_exs[i].words[j], add=True)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        index_vector = []
        for word in sentence:     
            index_vector.append(self.indexer.index_of(word))
        features = np.zeros([self.get_indexer_len()])
        for index in index_vector:
            features[index] = 1
        return features    
    
    
class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def get_indexer_len(self):
        return len(self.indexer)
    
    def set_train_indexer(self, train_exs):
        for i in range(len(train_exs)):
            for j in range(len(train_exs[i].words)):
                self.indexer.add_and_get_index(train_exs[i].words[j-1] + " " + train_exs[i].words[j], add=True)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        index_vector = []
        temp1 = sentence[0]
        for word in sentence:      
            if word == temp1: 
                continue
            index_vector.append(self.indexer.index_of(temp1 + " " + word))
            temp1 = word
        feature_vector = np.zeros([self.get_indexer_len()])
        for index in index_vector:
            feature_vector[index] = 1
        return feature_vector
    

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
         self.indexer = Indexer
         
    def set_train_indexer(self, train_exs):
        for i in range(len(train_exs)):
            for j in range(len(train_exs[i].words)):
                self.indexer.add_and_get_index(train_exs[i].words[j], add=True)

    def get_indexer(self):
        return self.indexer
    
    def get_indexer_len(self):
        return len(self.indexer)

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False)-> Counter:
        inp = sentence.split()
        return Counter([' '.join(gram) for n in sentence
                    for gram in zip(*[inp[i:] for i in range(n)])])


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, indexer: Indexer, extract_obj):
        unigram =  UnigramFeatureExtractor(indexer)
        indexer_len =  unigram.get_indexer_len()
        self.indexer = indexer
        self.extract = extract_obj
        self.w = np.zeros([indexer_len])
        self.train_ex = list()
        self.epoch = 0

    def set_traning_set(self, x):
        self.train_ex = x

    def set_epoch(self, epoch):
        self.epoch = epoch

    def perceptron(self):
        W = np.zeros([self.extract.get_indexer_len()])
        for ep in range(self.epoch):
            random.shuffle(self.train_ex)
            alpha = 1.0/self.epoch
            for ex in range(len(self.train_ex)):
                features = self.extract.extract_features(self.train_ex[ex].words)
                pred = np.dot(W, features) > 0
                if pred == self.train_ex[ex].label:
                    continue
                else:
                    if pred == 0 and self.train_ex[ex].label == 1:
                        W = W + features
                    else:
                        W = W - features
        self.w = W

    def predict(self, sentence: List[str]):
        features = self.extract.extract_features(sentence)
        if np.dot(self.w, features) > 0:
            return 1
        else:
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    
    def __init__(self, indexer: Indexer, extract_obj):
        unigram =  UnigramFeatureExtractor(indexer)
        indexer_len =  unigram.get_indexer_len()
        self.indexer = indexer
        self.extract = extract_obj
        self.w = np.zeros([indexer_len])
        self.train_ex = list()
        self.epoch = 0

    def set_traning_set(self, x):
        self.train_ex = x

    def set_epoch(self, epoch):
        self.epoch = epoch

    def LR(self):
        W = np.zeros([self.extract.get_indexer_len()])
        for ep in range(self.epoch):
            random.shuffle(self.train_ex)
            alpha = 1.0/self.epoch
            for ex in range(len(self.train_ex)):
                features = self.extract.extract_features(self.train_ex[ex].words)
                z = np.dot(W,features)
                W = W + alpha*features*(self.train_ex[ex].label-1.0/(1 + np.exp(-z)))
        self.w = W

    def predict(self, sentence: List[str]):
        features = self.extract.extract_features(sentence)
        if np.dot(self.w, features) > 0:
            return 1
        else:
            return 0

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    feat_extractor.set_train_indexer(train_exs)
    perceptron_model = PerceptronClassifier(train_exs, feat_extractor)
    perceptron_model.set_epoch(10)
    perceptron_model.set_traning_set(train_exs)
    perceptron_model.perceptron()
    indexer_len = feat_extractor.get_indexer_len()
    return perceptron_model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    feat_extractor.set_train_indexer(train_exs)
    LR_model = LogisticRegressionClassifier(train_exs, feat_extractor)
    LR_model.set_epoch(10)
    LR_model.set_traning_set(train_exs)
    LR_model.LR()
    indexer_len = feat_extractor.get_indexer_len()
    return LR_model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        for i in range(len(train_exs)):
            sentence = train_exs[i]
            sentence.words = [word for word in sentence.words if word != stopwords]
        for j in range(len(dev_exs)):
            sentence = dev_exs[j]
            sentence.words = [word for word in sentence.words if word != stopwords]
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model