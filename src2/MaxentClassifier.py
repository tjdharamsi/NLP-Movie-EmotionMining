import sys
from preprocess import InputInstance
import cPickle
from collections import defaultdict
from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from pycorenlp import StanfordCoreNLP
# import json
#from collections import OrderedDict
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.translate.ibm_model import Counts
#from numpy import array

class MaxentClassifier:
    
    def __init__(self):
        print('init')
        self.X_train = None
        self.y = None
        self.clf = None
        
    def createFeatureVectors(self, annData):
        print('createFeatureVectors')
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

        #----------Removes EmotionLesss-------
        key=[]
        for i in range(len(y_train)):
            if(y_train[i]=="emotionless"):
                key.append(i)
        
        AnnT=[]
        YT=[]
        for i in range(len(y_train)):
            if(i not in key):
                AnnT.append(annTokens[i])
                YT.append(y_train[i])
        
        annTokens=AnnT
        y_train=YT

        #-----------------------

        #print len(annTokens)
        ccounts = defaultdict(lambda: 0)
        for atlst in annTokens:
            for at in atlst:
                ccounts[at] += 1

        vlst = ccounts.keys()
        vlst.sort(key=lambda tup: tup[0])
        
#         print 'vlst: '
#         for v in vlst:
#             print v
        
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
        #RBF Kernel
        self.clf = svm.SVC( kernel="rbf",max_iter=1000, random_state=42,decision_function_shape='ovr')
        
        #Linear SVC
        #self.clf = svm.LinearSVC( max_iter=1000, random_state=42,multi_class='ovr')

        #self.clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=4, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    def crossvalidate(self):
        scores = cross_val_score(self.clf, self.X_train, self.y, cv=5,scoring="accuracy")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))        
        

if __name__ == '__main__':
    
    annData = None
    with open(sys.argv[1], 'rb') as f:
        annData = cPickle.load(f)
        
    classifier = MaxentClassifier()
    classifier.createFeatureVectors(annData)
    classifier.train()
    classifier.crossvalidate() 
