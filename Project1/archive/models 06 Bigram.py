# models.py

from multiprocessing.resource_sharer import stop
from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np 

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
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        #print("**Unigram: Extract_Feature is running")
        #print(self.indexer.get_object(0),self.indexer.get_object(1),self.indexer.get_object(2))
        #print(self.indexer.ints_to_objs)
        #exit()
        index_vector = []
        for word in sentence:     # sentence.words
            #print("unigram word:", word)
            #self.indexer.add_and_get_index("12 3")
            #index_vector.append(self.indexer.index_of("12 3"))
            index_vector.append(self.indexer.index_of(word))
        #print("index_vector",index_vector)
        feature_vector = np.zeros([self.vocab_size()])
        for index in index_vector:
            feature_vector[index] += 1
        return feature_vector    
    
    def vocab_size(self):
        return len(self.indexer)
    
    def build_vocab(self, train_exs):
        print("**Unigram: build_vocab is running")
        n = len(train_exs)
        #my_indexer = Indexer()
        for i in range(n):
            m = len(train_exs[i].words)
            for j in range(m):
                self.indexer.add_and_get_index(train_exs[i].words[j], add=True)
                #print(train_exs[i].words[j])
                #print(my_indexer.index_of(train_exs[i].words[j]))
            #exit()


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    def get_indexer(self):
        return self.indexer
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        #print("**Bigram: Extract_Feature is running")
        #print(self.indexer.get_object(0),self.indexer.get_object(1),self.indexer.get_object(2))
        #print(self.indexer.ints_to_objs)
        #exit()
        index_vector = []
        prev_word = sentence[0]
        for word in sentence:     # sentence.words
            if word == prev_word: continue
            #print("unigram word:", word)
            #self.indexer.add_and_get_index("12 3")
            #index_vector.append(self.indexer.index_of("12 3"))
            index_vector.append(self.indexer.index_of(prev_word + " " + word))
            prev_word = word
        #print("index_vector",index_vector)
        feature_vector = np.zeros([self.vocab_size()])
        for index in index_vector:
            feature_vector[index] += 1
        return feature_vector    
    
    def vocab_size(self):
        return len(self.indexer)
    
    def build_vocab(self, train_exs):
        print("**bigram: build_vocab is running")
        n = len(train_exs)
        #my_indexer = Indexer()
        for i in range(n):
            m = len(train_exs[i].words)
            for j in range(1,m):
                self.indexer.add_and_get_index(train_exs[i].words[j-1] + " " + train_exs[i].words[j], add=True)
                #print(train_exs[i].words[j])
                #print(my_indexer.index_of(train_exs[i].words[j]))
            #exit()

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


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
    def __init__(self, train_exs, feat_extractor):
    
    #def train_Perceptron(self, train_exs, feat_extractor):
        print("**PerceptronClassifier: init is running")
        #print("train_exs.words = ", train_exs[0].words)
        #print("train_exs[j] = ", train_exs[0])        
        self.feat_extractor = feat_extractor
        #traning the perceptron classifier
        self.feat_extractor.build_vocab(train_exs)
        n = len(train_exs)
        m = self.feat_extractor.vocab_size()
        weight_vector = np.zeros([m])
        Epoch = 10
        for i in range(Epoch):
            acc = np.zeros([n])
            for j in range(n):
                feat = self.feat_extractor.extract_features(train_exs[j].words)
                #print("n = ", n, " m = ", m)
                #print(feat.shape, "feat", feat)
                pred = float(np.dot(weight_vector,feat) > 0)
                #exit()
                if pred == train_exs[j].label:
                    acc[j] = 1
                    continue
                else:
                    if pred == 0 and train_exs[j].label == 1:
                        weight_vector = weight_vector + feat
                    else:
                        weight_vector = weight_vector - feat
            print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))   
        self.weight_vector = weight_vector

    def predict(self, sentence: List[str]):
        print("**PerceptronClassifier: predict is running")
        #print(self.feat_extractor.vocab_size())
        #print("sentence = ", sentence)
        #print("ex = ", ex.words)
        #self.feat_extractor.build_vocab(ex)
        feat = self.feat_extractor.extract_features(sentence)
        if np.dot(self.weight_vector, feat) > 0:
            print("prediction = 1")
            return 1
        else:
            print("prediction = 0")
            exit()
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    print("**train_perceptron is running")
    model = PerceptronClassifier(train_exs, feat_extractor)
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


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
    print("train model ---> train_exs[j] = ", train_exs[1])
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        print("**train_model: Unigram is running")

        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
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