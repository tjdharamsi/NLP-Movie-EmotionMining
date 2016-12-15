import numpy
import theano
import cPickle as pkl
from preprocess import InputInstance
from collections import defaultdict

class_dict1 = {'emotionless':0, 'happy':1, 'sad':2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6}
class_dict2 = {'happy':0, 'sad':1, 'surprise': 2, 'fear': 3, 'disgust': 4, 'anger': 5}
# dictionary for binary classification
class_dict3 = {'emotionless':0, 'happy':1, 'sad':1, 'surprise': 1, 'fear': 1, 'disgust': 1, 'anger': 1}

def grab_data(path, dictionary):
    subtitles = []
    classes = []
    with open(path, 'rb') as f:
        annData = pkl.load(f)

    for ii in xrange(len(annData)):
        tokens = []
        for s in annData[ii].atext['sentences']:
            tokens += [t['word'] for t in s['tokens']]
        subtitles.append(tokens)
        classes.append(annData[ii].label)
    seqs = [None] * len(subtitles)
    for idx, ss in enumerate(subtitles):
        #words = ss.strip().lower().split()
        words = ss
        #print 'words: ', words
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    # convert classes to indices
    classes = [class_dict1[cl] for cl in classes]
    #print classes
    
    return (seqs, classes)

# path to tokenized data, minlen, maxlen
def build_dict(path, minlen, maxlen):
    subtitles = []

    with open(path, 'rb') as f:
        annData = pkl.load(f)

    #annTokens = []
    # each subtitle may have multiple sentences so this one is used to compute average token length per sentence
    tokensPerSentence = []
    # each subtitle may have multiple sentences so this one is used to compute average token length per subtitle
    tokensPerSubtitle = []
    classes = []
    for ii in xrange(len(annData)):
        
        #atxt = json.loads(annData[ii].atext)
        tokens = []
        for s in annData[ii].atext['sentences']:
            tokensPerSentence.append(len(s['tokens'])) 
            tokens += [t['word'] for t in s['tokens']]
        tokensPerSubtitle.append(len(tokens))
        subtitles.append(tokens)
        classes.append(annData[ii].label)
    tokensPerSentence = numpy.asarray(tokensPerSentence)
    tokensPerSubtitle = numpy.asarray(tokensPerSubtitle)
    print 'average number of tokens per sentence: ', numpy.mean(tokensPerSentence)
    print 'min number of tokens per sentence: ', numpy.min(tokensPerSentence)
    print 'max number of tokens per sentence: ', numpy.max(tokensPerSentence)
    print 'average number of tokens per subtitle: ', numpy.mean(tokensPerSubtitle)
    print 'min number of tokens per subtitle: ', numpy.min(tokensPerSubtitle)
    print 'max number of tokens per subtitle: ', numpy.max(tokensPerSubtitle)        

    # print all subtitles less than minlen and more than maxlen and their corresponding label
    minlenSubtitlesIdx = [idx for idx in xrange(len(subtitles)) if len(subtitles[idx]) < minlen]
    maxlenSubtitlesIdx = [idx for idx in xrange(len(subtitles)) if len(subtitles[idx]) > maxlen]

    print 'Printing subtitles less than minlen ...'
    print 'Count of subtitles less than minlen: ', len(minlenSubtitlesIdx)
    minLenClassDist = defaultdict(lambda:0)
    for idx in minlenSubtitlesIdx:
        print 'id: ', idx
        print 'subtitle: ', subtitles[idx]
        print 'label: ', classes[idx]
        minLenClassDist[classes[idx]] += 1
    
    print 'maxLenClassDist: ', minLenClassDist.items()

    print 'Printing subtitles more than maxlen ...'
    print 'Count of subtitles more than maxlen: ', len(maxlenSubtitlesIdx)
    maxLenClassDist = defaultdict(lambda:0)
    for idx in maxlenSubtitlesIdx:
        print 'id: ', idx
        print 'subtitle: ', subtitles[idx]
        print 'label: ', classes[idx]
        maxLenClassDist[classes[idx]] += 1
        
    print 'maxLenClassDist: ', maxLenClassDist.items()

    print 'Building dictionary..',
    wordcount = dict()
    for ss in subtitles:
        #words = ss.strip().lower().split()
        words = ss
        #print 'words: ', words
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    # get indices in decreasing order of the counts by using [::-1]
    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()
    
    # the first value in the tuple returned by enumerate is 0-based index
    # worddict is a dictionary from words to indices such that words with higher count get a lower index compared to words with lower counts
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels

def load_data(path='', n_words=2000, valid_portion=0.1, test_portion=0.1, minlen=None, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset
 
    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknown (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.
 
    '''
 
    #############
    # LOAD DATA #
    #############
 
    # Load the dataset

    f = open(path, 'rb')
    data_set = pkl.load(f)
    #test_set = pickle.load(f)
    f.close()
    if minlen or maxlen:
        if minlen == None:
            minlen = 0
        if maxlen == None:
            maxlen = 100
        new_data_set_x = []
        new_data_set_y = []
        for x, y in zip(data_set[0], data_set[1]):
            if len(x) >= minlen and len(x) <= maxlen:
                new_data_set_x.append(x)
                new_data_set_y.append(y)
        data_set = (new_data_set_x, new_data_set_y)
        del new_data_set_x, new_data_set_y

    # split training set into validation set
    data_set_x, data_set_y = data_set
    n_samples = len(data_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion - test_portion)))
    n_valid = int(numpy.round(n_samples * valid_portion))
    print 'Total number of subtitles in ' + path, n_samples 
#     print 'Training set size: ', n_train
#     print 'Validation set size: ', n_valid
#     print 'Test set size: ', (n_samples - n_train - n_valid)
    train_set_x = [data_set_x[s] for s in sidx[:n_train]]
    train_set_y = [data_set_y[s] for s in sidx[:n_train]]
    valid_set_x = [data_set_x[s] for s in sidx[n_train:n_train + n_valid]]
    valid_set_y = [data_set_y[s] for s in sidx[n_train:n_train + n_valid]]
    test_set_x = [data_set_x[s] for s in sidx[n_train + n_valid:]]
    test_set_y = [data_set_y[s] for s in sidx[n_train + n_valid:]]
    
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
 
    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]
 
    #test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set
 
    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)
 
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
 
    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]
 
        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
 
        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
 
    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
 
    return train, valid, test


def main(path, dataset_name):
    #path = dataset_path
    print 'entering main...'
    dictionary = build_dict(path + dataset_name, 5, 20)
    X_y = grab_data(path + dataset_name, dictionary)
    f = open(path + dataset_name.rsplit('.',1)[0] + '.idx.pkl', 'wb')
    pkl.dump(X_y, f)
    f.close()

    f = open(path + dataset_name.rsplit('.',1)[0] + '.dict.pkl', 'wb')
    pkl.dump(dictionary, f)
    f.close()

if __name__ == '__main__':
    main('../data/', 'Titanic_dump.p')