# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np
import random 
import re
#import nltk
#from nltk.corpus import stopwords
#from multiprocessing.resource_sharer import stop

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
        #alpha = 1.0
        Epoch = 25
        for i in range(Epoch):
            acc = np.zeros([n])
            # shuffling the training set
            random.seed(Epoch + 1)
            random.shuffle(train_exs)
            alpha = 1.0/(Epoch + 5)
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
                        weight_vector = weight_vector + alpha*feat
                    else:
                        weight_vector = weight_vector - alpha*feat
            print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))   
        self.weight_vector = weight_vector

    def predict(self, sentence: List[str]):
        #print("**PerceptronClassifier: predict is running")
        #print(self.feat_extractor.vocab_size())
        #print("sentence = ", sentence)
        #print("ex = ", ex.words)
        #self.feat_extractor.build_vocab(ex)
        feat = self.feat_extractor.extract_features(sentence)
        if np.dot(self.weight_vector, feat) > 0:
            #print("prediction = 1")
            return 1
        else:
            #print("prediction = 0")
            #exit()
            #print("got wrong: ", sentence)
            #exit()
            return 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, train_exs, feat_extractor):
    #def train_LogisticReg(self, train_exs, feat_extractor):
        print("**LogisticRegressionClassifier: init is running")
        #print("train_exs.words = ", train_exs[0].words)
        #print("train_exs[j] = ", train_exs[0])        
        self.feat_extractor = feat_extractor
        #traning the Logistic Regression classifier
        self.feat_extractor.build_vocab(train_exs)
        n = len(train_exs)
        m = self.feat_extractor.vocab_size()
        weight_vector = np.zeros([m])
        #alpha = 1.0
        Epoch = 25
        for i in range(Epoch):
            acc = np.zeros([n])
            # shuffling the training set
            random.seed(Epoch + 1)
            random.shuffle(train_exs)
            alpha = 1.0/(Epoch + 5)
            for j in range(n):
                feat = self.feat_extractor.extract_features(train_exs[j].words)
                #print("n = ", n, " m = ", m)
                #print(feat.shape, "feat", feat)
                z = float(np.dot(weight_vector,feat))
                weight_vector = weight_vector + alpha*feat*(train_exs[j].label-self.sigmoid(z))
                #weight_vector = weight_vector + alpha*np.array(feat)*(train_exs[j].label-self.sigmoid(z))
                #acc[j] = - train_exs[j].label*np.log(self.sigmoid(z))-(1-train_exs[j].label)*np.log(1-self.sigmoid(z))
                if train_exs[j].label == 1: 
                    acc[j] = - np.log(self.sigmoid(z))
                else:    
                    acc[j] = - np.log(1-self.sigmoid(z))
                #acc[j] = np.log(1+np.exp(z)) - z
            print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))   
        self.weight_vector = weight_vector
        #exit()

    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def predict(self, sentence: List[str]):
        #print("**PerceptronClassifier: predict is running")
        #print(self.feat_extractor.vocab_size())
        #print("sentence = ", sentence)
        #print("ex = ", ex.words)
        #self.feat_extractor.build_vocab(ex)
        feat = self.feat_extractor.extract_features(sentence)
        if np.dot(self.weight_vector, feat) > 0:
            #print("prediction = 1")
            return 1
        else:
            #print("prediction = 0")
            #exit()
            return 0

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
    print("**train_logistic_regression is running")
    model = LogisticRegressionClassifier(train_exs, feat_extractor)
    return model


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
    #print("train model ---> train_exs[j] = ", train_exs[1])
    stopwords = [" ",".",",",":","'","/","*",";","...","-","--","'s","#","'ve","!","?","n't","'re","'ll","&","^","%","$","@","(",")"]
    #stopwords = set(stopwords.words('english'))
    function_words = ["a","aah","aboard","about","above","according","across","after","against","aged","ah","aha","alas","albeit","all",
    "along","alongside","although","am","amen","amid","amidst","among","amongst","an","and","another","any","anybody","anyone","anything",
    "are","aren","around","as","at","aye","back","be","because","been","before","behind","being","below","beneath","beside","besides",
    "between","beyond","billion","billionth","blah","both","but","by","bye","can","cannot","cheers","concerning","considering","cos",
    "could","couldn","crap","d","damn","dare","dear","despite","did","didn","do","does","doesn","doing","don","done","down","during",
    "each","eh","eight","eighteen","eighteenth","eighth","eightieth","eighty","either","eleven","eleventh","enough","every","everybody",
    "everyone","everything","except","excluding","farewell","few","fewer","fewest","fifteen","fifteenth","fifth","fiftieth","fifty",
    "first","five","following","for","former","fortieth","forty","four","fourteen","fourteenth","fourth","from","goddamn","goodbye",
    "goodnight","gosh","ha","had","hadn","half","has","hasn","have","haven","having","he","hello","her","hers","herself","hey","hi",
    "him","himself","his","hiya","hmm","ho","how","however","huh","hundred","hundredth","hurray","hush","i","if","immediately","in",
    "including","inside","into","is","isn","it","its","itself","last","latter","least","less","lest","like","little","ll","lots","m",
    "many","may","me","mhm","might","mightn","million","millionth","mine","minus","more","most","much","must","mustn","my","myself",
    "nah","nay","nd","near","need","needn","neither","next","nine","nineteen","nineteenth","ninetieth","ninety","ninth","no","nobody",
    "none","nope","nor","not","nothing","notwithstanding","of","off","oh","ok","okay","on","once","one","oneself","onto","ooh","oops",
    "opposite","or","ouch","ought","oughtn","our","ours","ourselves","out","outside","over","own","past","pending","per","plenty","plus",
    "provided","providing","rd","re","regarding","right","round","s","same","second","seven","seventeen","seventeenth","seventh",
    "seventieth","seventy","several","shall","shalt","shan","she","should","shouldn","since","six","sixteen","sixteenth","sixth",
    "sixtieth","sixty","so","some","somebody","someone","something","st","such","supposing","t","ten","tenth","th","than","that","the",
    "thee","their","theirs","them","themselves","there","these","they","third","thirteen","thirteenth","thirtieth","thirty","this",
    "those","thou","though","thousand","thousandth","three","through","throughout","thru","thy","til","till","to","toward","towards",
    "trillion","trillionth","twelfth","twelve","twentieth","twenty","two","uh","um","under","underneath","unless","unlike","until",
    "unto","up","upon","urgh","us","used","ve","versus","via","vice","vs","was","wasn","we","well","were","weren","what","whatever",
    "whatsoever","when","whenever","where","whereas","whereupon","wherever","whether","which","whichever","while","whilst","who",
    "whoever","whom","whose","why","will","with","within","without","won","worth","would","wouldn","wow","ye","yeah","yep","yes","you",
    "your","yours","yourself","yourselves","yum","zero"]
    stopwords.append(function_words)
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        print("**train_model: Unigram is running")
        #print("train1: ", train_exs[1])
        for i in range(len(train_exs)):
            sentence = train_exs[i]
            #sentence.words = [re.sub(r'[^a-zA-Z]', '', word).strip().lower() for word in sentence.words if word.lower() not in chain(stop_words, niose))]
            sentence.words = [re.sub(r'[^a-zA-Z]', '', word).strip().lower() for word in sentence.words if len(word)>3 and word.lower() not in stopwords]
        #print("train2: ", train_exs[1])
        #exit()
        for i in range(len(dev_exs)):
            sentence = dev_exs[i]
            #sentence.words = [re.sub(r'[^a-zA-Z]', '', word).strip().lower() for word in sentence.words if word.lower() not in chain(stop_words, niose))]
            sentence.words = [re.sub(r'[^a-zA-Z]', '', word).strip().lower() for word in sentence.words if len(word)>3 and word.lower() not in stopwords]
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        print("**train_model: Bigram is running")
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