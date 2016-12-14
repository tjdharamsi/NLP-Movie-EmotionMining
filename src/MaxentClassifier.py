import sys
from preprocess import InputInstance
import cPickle
from collections import defaultdict
from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.preprocessing import normalize
from numpy import dtype
from nltk.util import ngrams
import json
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import classification_report

other_features_dict = {'Titanic': '../nitesh_features/Titanic_features.json', 'combined': '../nitesh_features/combined_features.json',
                       'Friends': '../nitesh_features/Friends_features.json', 'Walking_Dead': '../nitesh_features/Walking_Dead_features.json' }

#from pycorenlp import StanfordCoreNLP
# import json
#from collections import OrderedDict
#from sklearn.feature_extraction.text import CountVectorizer
#from nltk.translate.ibm_model import Counts
#from numpy import array
class_dict = {'emotionless':0, 'happy':1, 'sad':2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6}
rev_class_dict={0:'emotionless',1:'happy',2: 'sad',3: 'surprise', 4:'fear', 5:'disgust',6: 'anger'}

class MaxentClassifier:
    
    def __init__(self, ignoreEmotionLess=False):
        print('init')
        self.X_train = None
        self.y = None
        self.clf = None
        self.wordToIdx = None
        self.IdxToWord = None
        self.topFeatures = None
        self.other_features = None
        self.ignoreEmotionLess = ignoreEmotionLess


    def createFeatureVectors(self, annData):
        print('createFeatureVectors')
        annTokens = []
        y_train = []
        for ii in xrange(len(annData)):
            #atxt = json.loads(annData[ii].atext)
            tokens = []
            allTokens = []
            #pos_tags = []
            for s in annData[ii].atext['sentences']:
                tokens += [t['word'].lower() for t in s['tokens'] if t['pos'] in ('JJ', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN')]
                allTokens += [t['word'].lower() for t in s['tokens']]
                #pos_tags += [t['pos'] for t in s['tokens'] if t['pos'] in ('JJ', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN')]
            bigrams = ngrams(allTokens,2)
            tokens =    list(bigrams)+tokens
            #print tokens
            annTokens.append(tokens)
            #annTokens.append(bigrams)
            y_train.append(annData[ii].label)

        #self.other_features
        # remove emotionless class
        emotionless_indices=[]
        if self.ignoreEmotionLess:
            for i in range(len(y_train)):
                if(y_train[i]=="emotionless"):
                    emotionless_indices.append(i)
            
            AnnT=[]
            YT=[]
            X_train_indices = []    # keeps track of the indices of the considered subtitles which have class other than emotionless
            for i in range(len(y_train)):
                if(i not in emotionless_indices):
                    AnnT.append(annTokens[i])
                    YT.append(y_train[i])
                    X_train_indices.append(i+1)
            
            annTokens=AnnT
            y_train=YT
        
        
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
        
        V = np.asarray(V, dtype=np.float64)
        
        X_train = sparse.coo_matrix((V,(I,J)),shape=(len(annTokens),len(vocabulary))).tocsr()
        X_train = normalize(X_train, norm='l1', axis=1)
        
        # add other features over here
        X_train = X_train.toarray()
        print(len(self.other_features))
        print(X_train.shape[0])
        assert (len(self.other_features) - len(emotionless_indices)) == X_train.shape[0]
        assert (len(X_train_indices) == X_train.shape[0])
        
        extra_features = []
        for ii in range(X_train.shape[0]):
            #print ii
            #print self.other_features[str(ii)]
            
            # append previous labels
            
            key1 = None
            key2 = None
            key3 = None
            
            if self.other_features[str(X_train_indices[ii])]['prev1_emotion'] == 0:
                key1 = 'emotionless'
            else:
                key1 = self.other_features[str(X_train_indices[ii])]['prev1_emotion']
 
            if self.other_features[str(X_train_indices[ii])]['prev2_emotion'] == 0:
                key2 = 'emotionless'
            else:
                key2 = self.other_features[str(X_train_indices[ii])]['prev2_emotion']
 
            if self.other_features[str(X_train_indices[ii])]['prev3_emotion'] == 0:
                key3 = 'emotionless'
            else:
                key3 = self.other_features[str(X_train_indices[ii])]['prev3_emotion']
             
#             print class_dict[key1]
#             print class_dict[key2]
#             print class_dict[key3]
            
            prev_label1 = np.zeros(7)
            prev_label1[class_dict[key1]] = 1
            
            prev_label2 = np.zeros(7)
            prev_label2[class_dict[key2]] = 1

            prev_label3 = np.zeros(7)
            prev_label3[class_dict[key3]] = 1
            
            # add punctuation features
            punc_features = [self.other_features[str(X_train_indices[ii])]['eight_note_mark'], self.other_features[str(X_train_indices[ii])]['exclamation_pt'], 
                             self.other_features[str(X_train_indices[ii])]['question_mark']]

            
            # add pos tag percentages
            POS_features=[]
            for tag in ['NN','VB','JJ','ADV']:
                POS_features.append(self.other_features[str(X_train_indices[ii])][tag+"_percent"])



            
            
            
            #extra_fv = np.concatenate((prev_label1,prev_label2,prev_label3, POS_features, punc_features))
            #extra_fv = np.concatenate((prev_label2, prev_label3, POS_features,punc_features))
            extra_fv = np.concatenate((punc_features,POS_features,prev_label3,prev_label2,prev_label1))
            
            #extra_fv=np.concatenate((punc_features, POS_features,prev_label3,prev_label2))
            #extra_fv=punc_features

            

            
            
            extra_features.append(extra_fv)
            
        
        extra_features = np.asarray(extra_features)
        
        
         
        X_train = np.concatenate((X_train, extra_features), axis=1)
        #X_train=extra_features
        print('Feature space dimensionality: ', X_train.shape[1])
         
        #print(X_train.shape)

        self.y = [class_dict[cl] for cl in y_train]
        
        #print set(self.y)
        
        self.X_train = X_train
        #self.y = np.asarray(self.y)
        #print(self.y)
    
    def train(self):
#         
        #self.clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42,multi_class='ovr')
        self.clf = LogisticRegression(max_iter=1000, random_state=42,multi_class='ovr')

        #RBF Kernel
        #self.clf = svm.SVC( kernel="rbf",max_iter=1000, random_state=42,decision_function_shape='ovr')

        
        #Linear SVC
        self.clf = svm.LinearSVC( max_iter=1000, random_state=42,multi_class='ovr')

        #RandomForest
        #self.clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=4, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

    def crossvalidate(self):
        scores = cross_val_score(self.clf, self.X_train, self.y, cv=5)
        predicted = cross_val_predict(self.clf,self.X_train,self.y, cv=5)
        #print(set(predicted))
        keerti_results=[]
        for i in predicted:
            keerti_results.append(rev_class_dict[i])
        #print(keerti_results)
        f=open('results.txt','w')
        for i in keerti_results:
            f.write(i+"\n")
        f.close()
        count=0
        for i in (predicted):
            if(i==2):
                count+=1
        #print(count)
        class_names=["emotionless","happy","sad","surprise","fear","disgust","anger"]
        """
        plt.figure()
        cnf_matrix = confusion_matrix(self.y, predicted)
        np.set_printoptions(precision=2)
        if self.ignoreEmotionLess:
            class_names=["happy","sad","surprise","fear","disgust","anger"]
        else:
            class_names=["emotionless","happy","sad","surprise","fear","disgust","anger"]
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix, without normalization')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()"""
        
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #print(classification_report(self.y, predicted, target_names=class_names))

    def validate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y, test_size=0.2, random_state=42)
        print(self.clf.fit(X_train, y_train).score(X_test,y_test))
        ans=self.clf.predict(X_test)
        #print(ans)
        y_train.extend(ans)
        #print(y_train)
        keerti_results=[]
        for i in y_train:
            keerti_results.append(rev_class_dict[i])
        #print(keerti_results)
        f=open('results.txt','w')
        for i in keerti_results:
            f.write(i+"\n")
        f.close()
        
        #Confusion Matrix
        plt.figure()
        cnf_matrix = confusion_matrix(y_test, ans)
        np.set_printoptions(precision=2)
        if self.ignoreEmotionLess:
            class_names=["happy","sad","surprise","fear","disgust","anger"]
        else:
            class_names=["emotionless","happy","sad","surprise","fear","disgust","anger"]
        plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix, without normalization')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        thresh = cnf_matrix.max() / 2.
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()







    def getTopFeatures(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y, test_size=0.2, random_state=42)
        selector = RFE(self.clf, 50, step=1)
        selector = selector.fit(X_train, y_train)
        ranking_ = selector.ranking_
        self.topFeatures = [self.IdxToWord[idx] for idx in xrange(len(ranking_)) if ranking_[idx] == 1]
        print(self.topFeatures)
        
    def readOtherFeatures(self, ofFile):
        with open(ofFile, 'rb') as f:
            self.other_features = json.load(f)
        #print(self.other_features)

    



if __name__ == '__main__':
    
    annData = None
    with open(sys.argv[1], 'rb') as f:
        annData = cPickle.load(f)
        
<<<<<<< Updated upstream
    classifier = MaxentClassifier(ignoreEmotionLess=True)
    classifier.readOtherFeatures(other_features_dict['combined'])
    #classifier.createFeatureVectors(annData)
=======
    classifier = MaxentClassifier()
    classifier.readOtherFeatures(other_features_dict['Walking_Dead'])
>>>>>>> Stashed changes
    classifier.createFeatureVectors(annData)
    classifier.train()
    classifier.crossvalidate()
    #classifier.validate()
    #classifier.getTopFeatures()