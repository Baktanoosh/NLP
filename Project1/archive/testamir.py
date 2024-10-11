import numpy as np

def perceptron(train_exs, label):    
    Epoch = 10
    n = len(train_exs)
    m = len(train_exs[0])
    weight_vector = np.zeros([m])    
    for i in range(Epoch):
        print("-------epoch = ", i,"---------------------")
        acc = np.zeros([n])
        for j in range(n):
            print("------- data = ", j)
            feat = train_exs[j]
            pred = float(np.dot(weight_vector,feat) > 0)
            print("feat", feat)
            print("weight", weight_vector)
            print("w_f", pred, float(np.dot(weight_vector,feat)))
            print("label", label[j])
            if pred == label[j]:
                acc[j] = 1
                continue
            else:
                if pred == 0 and label[j] == 1:
                    weight_vector = weight_vector + feat
                else:
                    weight_vector = weight_vector - feat
            print("new w",  weight_vector)
            print("loss = ", pred*label[j])
        print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))   
        #exit()
    return weight_vector

train_exs = [[1,0,2,0,1,3],[0,0,1,1,0,0],[2,2,2,0,0,0]]
label = [1,0,1]

perceptron(train_exs, label)

"""
from models import *
from utils import *

class Unigram():
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    def get_indexer(self):
        return self.indexer
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        print("**Unigram: Extract_Feature is running")
        index_vector = []
        for word in sentence:     # sentence.words
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
            #m = len(train_exs[i].words)
            m = len(train_exs[i])
            for j in range(m):
                #self.indexer.add_and_get_index(train_exs[i].words[j], add=True)
                self.indexer.add_and_get_index(train_exs[i][j], add=True)
                #print(train_exs[i].words[j])
                #print(my_indexer.index_of(train_exs[i].words[j]))
            #exit()

unif = Unigram(Indexer)

def preceptron(train_exs, label):    
    Epoch = 10
    for i in range(Epoch):
        unif.build_vocab(train_exs)
        n = len(train_exs)
        m = unif.vocab_size()
        weight_vector = np.zeros([m])
        acc = np.zeros([n])
        for j in range(n):
            feat = unif.extract_features(train_exs[j])
            #print("n = ", n, " m = ", m)
            #print(feat.shape, "feat", feat)
            pred = float(np.dot(weight_vector,feat) > 0)
            print("loss = ", pred*label[j])
            #exit()
            if pred == label[j]:
                acc[j] = 1
                continue
            else:
                if pred == 0 and train_exs[j].label == 1:
                    weight_vector = weight_vector + feat
                else:
                    weight_vector = weight_vector - feat
        print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))   
        exit()
    return weight_vector

train_exs = [['The', 'Rock', 'is', 'destined', 'to', 'be'],['The', 'gorgeously', 'elaborate', 'continuation', 'of'],['Rings', "''", 'trilogy', 'is', 'so', 'huge', 'that', 'a', 'column', 'of', 'words']]
label = [1,0,1]

preceptron(train_exs, label)
"""

"""
from models import *
#UnigramFeatureExtractor()

import numpy as np 
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

input = "The orange is the fruit of the citrus species Citrus x sinensis in the family Rutaceae. It is also called sweet orange, to distinguish it from the related Citrus x aurantium, referred to as bitter orange. The sweet orange reproduces asexually (apomixis through nucellar embryony); varieties of sweet orange arise through mutations. The orange is a hybrid between pomelo (Citrus maxima) and mandarin (Citrus reticulata). It has genes that are ~25% pomelo and ~75% mandarin; however, it is not a simple backcrossed BC1 hybrid, but hybridized over multiple generations. The chloroplast genes, and therefore the maternal line, seem to be pomelo. The sweet orange has had its full genome sequenced. Earlier estimates of the percentage of pomelo genes varying from ~50% to 6% have been reported. Sweet oranges were mentioned in Chinese literature in 314 BC. As of 1987, orange trees were found to be the most cultivated fruit tree in the world. Orange trees are widely grown in tropical and subtropical climates for their sweet fruit. The fruit of the orange tree can be eaten fresh, or processed for its juice or fragrant rind. As of 2012, oranges accounted for approximately 70% of citrus production. In 2014, 70.9 million tonnes of oranges were grown worldwide, with Brazil producing 24% of the world total followed by China and India."
segments = input.split('ee')
length = []
segment_length = [len(segment) for segment in segments]

idx1 = np.argmax(segment_length)

print(len(segments[idx1]))
print('longest segment: %s' % segments[idx1])
for segment in segments:
  length.append(len([ii for ii in find_all(segment, 'th')]))
idx2 = np.argmax(length)
#import ipdb;ipdb.set_trace()
print('maximum th times %d' % length[idx2])
cond = (idx1 == idx2)
print('%d' % cond)

"""
