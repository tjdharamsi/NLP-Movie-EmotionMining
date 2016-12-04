import sys
from preprocess import InputInstance
import cPickle
from collections import defaultdict
from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
#from pycorenlp import StanfordCoreNLP
# import json
#from collections import OrderedDict
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.translate.ibm_model import Counts
#from numpy import array

class MaxentClassifier:
    
    def __init__(self):
        print 'init'
        self.X_train = None
        self.y = None
        self.clf = None
        self.wordToIdx = None
        self.IdxToWord = None
        self.topFeatures = None
        
    def createFeatureVectors(self, annData):
        print 'createFeatureVectors'
        annTokens = []
        y_train = []
        for ii in xrange(len(annData)):
            #atxt = json.loads(annData[ii].atext)
            tokens = []
            for s in annData[ii].atext['sentences']:
                tokens += [(t['word'].lower(), t['pos']) for t in s['tokens'] if t['pos'] in ('JJ', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN')]
            #print tokens
            annTokens.append(tokens)
            y_train.append(annData[ii].label)

        # remove emotionless class            
#         key=[]
#         for i in range(len(y_train)):
#             if(y_train[i]=="emotionless"):
#                 key.append(i)
#         
#         AnnT=[]
#         YT=[]
#         for i in range(len(y_train)):
#             if(i not in key):
#                 AnnT.append(annTokens[i])
#                 YT.append(y_train[i])
#         
#         annTokens=AnnT
#         y_train=YT

        # we get the feature space below
        ccounts = defaultdict(lambda: 0)
        for atlst in annTokens:
            for at in atlst:
                ccounts[at] += 1

        vlst = ccounts.keys()
        vlst.sort(key=lambda tup: tup[0])
        
        vocabulary = defaultdict()
        for ii in xrange(len(vlst)):
            #print 'vlst[ii]', vlst[ii]
            vocabulary[vlst[ii]] = ii
        
#         for k, v in vocabulary.items():
#             print k, v

#         ccounts = OrderedDict()
#         for atlst in annTokens:
#             for at in atlst:
#                 if at in ccounts:
#                     ccounts[at] += 1
#                 else:
#                     ccounts[at] = 1        

        self.wordToIdx = vocabulary
        print 'Feature space dimensionality: ', len(self.wordToIdx)
        
        # reverse index to obtain idx to word
        self.IdxToWord = {v: k for k, v in self.wordToIdx.iteritems()}
        
        V = []
        I = []
        J = []
        for ii in xrange(len(annTokens)):
            tmpd = defaultdict(lambda: 0)
            for at in annTokens[ii]:
                tmpd[at] += 1
            for key in tmpd:
                V.append(tmpd[key])
                I.append(ii)
                J.append(vocabulary[key])
        
        X_train = sparse.coo_matrix((V,(I,J)),shape=(len(annTokens),len(vocabulary)))
        labels = defaultdict()
        for ii in xrange(len(y_train)):
            labels[y_train[ii]] = ii
        y = [labels[y_i] for y_i in y_train]
        
        self.X_train = X_train
        self.y = np.asarray(y)
    
    def train(self):
        self.clf = LogisticRegression(solver='sag', max_iter=1000, random_state=42,
                             multi_class='ovr')
    def crossvalidate(self):
        scores = cross_val_score(self.clf, self.X_train, self.y, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 
        
    def getTopFeatures(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y, test_size=0.2, random_state=42)
        selector = RFE(self.clf, 50, step=1)
        selector = selector.fit(X_train, y_train)
        ranking_ = selector.ranking_
        self.topFeatures = [self.IdxToWord[idx] for idx in xrange(len(ranking_)) if ranking_[idx] == 1]
        print self.topFeatures

if __name__ == '__main__':
    
    annData = None
    with open(sys.argv[1], 'rb') as f:
        annData = cPickle.load(f)
        
    classifier = MaxentClassifier()
    classifier.createFeatureVectors(annData)
    classifier.train()
    classifier.crossvalidate()
    classifier.getTopFeatures()