
import json
import scipy
from scipy.stats import pearsonr


other_features_dict = {'Titanic': '../nitesh_features/Titanic_features.json', 'combined': '../nitesh_features/combined_features.json',
                       'Friends': '../nitesh_features/Friends_features.json', 'Walking_Dead': '../nitesh_features/Walking_Dead_features.json' }

data_set_labels_files = {'Titanic': '../nitesh_features/Titanic.emotion', 'combined': '../nitesh_features/combined.emotion',
                       'Friends': '../nitesh_features/friends.emotion', 'Walking_Dead': '../nitesh_features/wd.emotion' }


data_sets = ['Titanic', 'Friends', 'Walking_Dead', 'combined']
class_dict = {'emotionless':0, 'happy':1, 'sad':2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6}

features = ["eight_note_mark", "exclamation_pt", "question_mark"]
class_labels = [0,1,2,3,4,5,6]

for ds in data_sets:
    
    print '-' * 50
    print 'data set: ', ds
    
    other_features = None
    ds_labels = []
    with open(other_features_dict[ds], 'rb') as f:
        other_features = json.load(f)
    with open(data_set_labels_files[ds], 'rb') as f:
        for line in f:
            ds_labels.append(class_dict[line.strip()])
    
    assert len(other_features) == len(ds_labels)
    
    fcorrv = {"eight_note_mark": [], "exclamation_pt": [], "question_mark": []}
    labelcorrv = {0:[], 1:[], 2:[], 3: [], 4: [], 5: [], 6: []}
    for ii in xrange(len(ds_labels)):
        for feature in features:
            fcorrv[feature].append(other_features[str(ii + 1)][feature])
        
        for label in class_labels:
            if label == ds_labels[ii]:
                labelcorrv[label].append(1.0)
            else:
                labelcorrv[label].append(0.0)
                
    for label in class_labels:
        for feature in features:
            assert len(labelcorrv[label]) == len(fcorrv[feature])
            print 'label: ', label
            print 'feature: ', feature
            print 'pearson correlation: ', pearsonr(fcorrv[feature], labelcorrv[label])
            
    print '-' * 50