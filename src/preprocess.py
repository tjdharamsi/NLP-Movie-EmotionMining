from pycorenlp import StanfordCoreNLP
import sys, getopt
import cPickle
import json
import re
import nltk
import itertools
import numpy

class InputInstance:
    def __init__(self, i, ts, text, atext=None, label=None):
        self.i = i
        self.ts = ts
        self.text = text
        self.atext = atext
        self.label = label
        
    def __str__(self):
        iidict = {}
        iidict['i'] = self.i
        iidict['ts'] = self.ts
        iidict['text'] = self.text
        iidict['atext'] = self.atext
        iidict['label'] = self.label
        return json.dumps(iidict)
        
class PreProcessor:
    
    def __init__(self):
        print 'Initializing PreProcessor'

    def annotateAll(self, inputData):
        nlp = StanfordCoreNLP('http://localhost:9000')
        for inp in inputData:
            #inp.atext = json.dumps(nlp.annotate(inp.text, properties={'annotators': 'tokenize, ssplit, pos, lemma, ner','outputFormat': 'json'}))
            inp.atext = nlp.annotate(inp.text, properties={'annotators': 'tokenize, ssplit, pos, lemma, ner','outputFormat': 'json'})
            #print inp.atext
        
    # method to annotate a single sentence for quick testing
    def annotateText(self, text):
        nlp = StanfordCoreNLP('http://localhost:9000')
        return nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos','outputFormat': 'json'})
    
    def readInput(self, inputXFile, inputYFile):
        inputData = []
        with open(inputXFile,'r') as f:
            currInput = []
            for line in f:
                line = line.strip()
#                 print 'len(line): ', len(line)
                if len(line) > 0:
                    #print line
                    #print line.encode('utf-8')
                    line = re.sub(r'[^\x00-\x7F]+',' ', line)
                    #print line
                    currInput.append(line)
                else:
                    i = currInput[0]
#                     print '###############'
#                     print i
#                     print '###############'                    
                    ts = currInput[1]
#                     print '###############'
#                     print ts
#                     print '###############'
                    text = ' '.join(currInput[2:])
                    #print text
                    #text = re.sub(u'[^\u0000-\u007e]+',' ', text)
                    #re.sub(r'[^\x00-\x7F]+',' ', text)
#                     print '###############'
#                     print text
#                     print '###############'                    
                    inputData.append(InputInstance(i, ts, text))
                    currInput = []
        if(len(currInput) >= 3):
            inputData.append(InputInstance(currInput[0], currInput[1], ' '.join(currInput[2:])))
        
        with open(inputYFile, 'r') as f:
            labels = f.readlines()
            assert (len(labels) == len(inputData)),'number of labels not equal to number of inputs'
            ii = 0
            for l in labels:
                inputData[ii].label = l.strip()
                ii += 1
        
#         for idata in inputData:
#             print idata
            
        return inputData

    # returns three dictionaries, 1st: wordToIndex, 2nd: labelToIndex, 3rd: frequency distribution (word to count)
    def getDictionaries(self, annData):
        annTokens = []
        # each subtitle may have multiple sentences so this one is used to compute average token length per sentence
        tokensPerSentence = []
        # each subtitle may have multiple sentences so this one is used to compute average token length per subtitle
        tokensPerSubtitle = []
        y_train = []
        for ii in xrange(len(annData)):
            #atxt = json.loads(annData[ii].atext)
            tokens = []
            for s in annData[ii].atext['sentences']:
                tokensPerSentence.append(len(s['tokens'])) 
                tokens += [t['word'] for t in s['tokens']]
            tokensPerSubtitle.append(len(tokens))
            annTokens.append(tokens)
            y_train.append(annData[ii].label)
        tokensPerSentence = numpy.asarray(tokensPerSentence)
        tokensPerSubtitle = numpy.asarray(tokensPerSubtitle)
        word_freq = nltk.FreqDist(itertools.chain(*annTokens))
        vocab = word_freq.most_common()
        index_to_word = [x[0] for x in vocab]
        #index_to_word.append(unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
        class_freq = nltk.FreqDist(y_train)
        classes = class_freq.most_common()
        index_to_class = [x[0] for x in classes]
        class_to_index = dict([(w,i) for i,w in enumerate(index_to_class)])
        print 'average number of tokens per sentence: ', numpy.mean(tokensPerSentence)
        print 'min number of tokens per sentence: ', numpy.min(tokensPerSentence)
        print 'max number of tokens per sentence: ', numpy.max(tokensPerSentence)
        print 'average number of tokens per subtitle: ', numpy.mean(tokensPerSubtitle)
        print 'min number of tokens per subtitle: ', numpy.min(tokensPerSubtitle)
        print 'max number of tokens per subtitle: ', numpy.max(tokensPerSubtitle)        
        print 'vocabulary size: ', len(vocab)
        return (word_to_index, class_to_index, vocab)

    def preprocess(self, inputXFile, inputYFile):
        print 'preprocessing data...'
        inputData = self.readInput(inputXFile, inputYFile)
        self.annotateAll(inputData)
        word_to_index, class_to_index, vocab = self.getDictionaries(inputData)
        return (inputData, word_to_index, class_to_index, vocab)

if __name__ == '__main__':
    inputXFile = None
    inputYFile = None
    dumpFile = None       
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'h', ['help', 'xfile=', 'yfile=', 'dfile='])
    except getopt.GetoptError as err:
        print(err)
        print 'preprocess.py --xfile <x_file> --yfile <yte_file> --dfile <dump_file>'
        sys.exit(2)
    for opt, arg in options:
        if opt in ('-h', '-help'):
            print 'preprocess.py --xfile <x_file> --yfile <yte_file> --dfile <dump_file>'
            sys.exit()
        elif opt == '--xfile':
            inputXFile = arg
        elif opt == '--yfile':
            inputYFile = arg
        elif opt == '--dfile':
            dumpFile = arg        
        else:
            print 'unhandled option'
    
    print 'inputXFile: ', inputXFile
    print 'inputYFile: ', inputYFile
    pp = PreProcessor()
    ainputData = pp.preprocess(inputXFile, inputYFile)
#     if dumpFile is not None:
#         with open(dumpFile, 'wb') as fp:
#             cPickle.dump(ainputData, fp)