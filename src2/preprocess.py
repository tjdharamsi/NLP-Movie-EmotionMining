from pycorenlp import StanfordCoreNLP
import sys, getopt
import cPickle
import json
import re

class InputInstance:
    def __init__(self, i, ts, text):
        self.i = i
        self.ts = ts
        self.text = text
        self.atext = None        
        self.label = None
        
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
        print('Initializing PreProcessor')

    def annotateAll(self, inputData):
        nlp = StanfordCoreNLP('http://localhost:9000')
        for inp in inputData:
            #inp.atext = json.dumps(nlp.annotate(inp.text, properties={'annotators': 'tokenize, ssplit, pos, lemma, ner','outputFormat': 'json'}))
            inp.atext = nlp.annotate(inp.text, properties={'annotators': 'tokenize, ssplit, pos, lemma, ner','outputFormat': 'json'})
            #print inp.atext
            
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

    def preprocess(self, inputXFile, inputYFile):
        print('preprocessing data...')
        inputData = self.readInput(inputXFile, inputYFile)
        self.annotateAll(inputData)
        return inputData

if __name__ == '__main__':
    inputXFile = "friends.srt"
    inputYFile = "friends.emotion"
    dumpFile = "dump2"       
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'h', ['help', 'xfile=', 'yfile=', 'dfile='])
    except getopt.GetoptError as err:
        print(err)
        print('preprocess.py --xfile <x_file> --yfile <yte_file> --dfile <dump_file>')
        sys.exit(2)
    for opt, arg in options:
        if opt in ('-h', '-help'):
            print('preprocess.py --xfile <x_file> --yfile <yte_file> --dfile <dump_file>')
            sys.exit()
        elif opt == '--xfile':
            inputXFile = arg
        elif opt == '--yfile':
            inputYFile = arg
        elif opt == '--dfile':
            dumpFile = arg        
        else:
            print('unhandled option')
    
    print('inputXFile: ', inputXFile)
    print('inputYFile: ', inputYFile)
    pp = PreProcessor()
    ainputData = pp.preprocess(inputXFile, inputYFile)
    if dumpFile is not None:
        with open(dumpFile, 'wb') as fp:
            cPickle.dump(ainputData, fp)