import nltk
import pprint
import json
from nltk.stem import WordNetLemmatizer
from string import punctuation
from gi.overrides.keysyms import exclam
wordnet_lemmatizer = WordNetLemmatizer()

other_features_dict = {'Titanic': '../nitesh_features/Titanic_features.json', 
					   'Friends': '../nitesh_features/Friends_features.json', 'Walking_Dead': '../nitesh_features/Walking_Dead_features.json' }

eight_note = u'\u266a'

LEXICON = {}
# Load Lexicon
lex = open("lexicon.csv")
for line in lex:
	dic = {"Positive":0,"Negative":0,"Anger":0,"Anticipation":0,"Disgust":0,"Fear":0,"Joy":0,"Sadness":0,"Surprise":0,"Trust":0}
	field = line.split(",")
	LEXICON[field[0]] = dic
	dic['Positive'] = int(field[1])
	dic['Negative'] = int(field[2])
	dic['Anger'] = int(field[3])
	dic['Anticipation'] = int(field[4])
	dic['Disgust'] = int(field[5])
	dic['Fear'] = int(field[6])
	dic['Joy'] = int(field[7])
	dic['Sadness'] = int(field[8])
	dic['Surprise'] = int(field[9])
	dic['Trust'] = int(field[10])

lexicon_keys = LEXICON.keys()

EMOTIONS = {}
i = 1
# Read emotion file
fh = open("friends.emotion")
for line in fh:
	line = line.strip()
	EMOTIONS[i] = line
	i += 1

f = open("friends.srt")

k = 0
POS = ['NN','VB','JJ','RB']
pp = pprint.PrettyPrinter(indent=4)

		
found = 0
not_found = 0

FEATURES = {}

i = 0
for line in f:
	if k == 0:
		i += 1

		FEATURES[i] = {"NN_percent" : 0, "VB_percent" : 0, "JJ_percent" : 0, "ADV_percent":0, "anger_prob" : 0, "disgust_prob":0, "emotionless_prob":0, "fear_prob":0, "happy_prob":0, "sad_prob":0, "surprise_prob":0, "prev1_emotion":0, "prev2_emotion" :0, "prev3_emotion":0, "eight_note_mark":0, "exclamation_pt":0, "question_mark":0}

	elif k > 1:
		# Finding POS percents
		line = line.decode("utf-8").strip()
		tokens = nltk.word_tokenize(line)
		pos_tag = nltk.pos_tag(tokens)

		nn_count = 0
		jj_count = 0
		vb_count = 0
		rb_count = 0
		total = 0

		for pos in pos_tag:
			tag = pos[1][0:2]
			# print tag
			if tag == "NN":
				nn_count += 1
			
			elif tag == "VB":
				vb_count += 1

			elif tag == "JJ":
				jj_count += 1

			elif tag == "RB":
				rb_count += 1

			if tag in POS:
				total += 1

		if total > 0:
			FEATURES[i]["NN_percent"] = nn_count * 1.0/total
			FEATURES[i]["JJ_percent"] = jj_count * 1.0/total
			FEATURES[i]["ADV_percent"] = rb_count * 1.0/total
			FEATURES[i]["VB_percent"] = vb_count * 1.0/total


		# find out if special characters are present in the subtitle

		if '?' in line:
			FEATURES[i]["question_mark"] = 1

		if '!' in line:
			FEATURES[i]["exclamation_pt"] = 1
		
		if eight_note in line:
			FEATURES[i]["eight_note_mark"] = 1
		
		# Finding EMOTION percent
		emotion = {"anger_prob" : 0, "disgust_prob":0, "emotionless_prob":0, "fear_prob":0, "happy_prob":0, "sad_prob":0, "surprise_prob":0}
			
		for pos in pos_tag:
			word = pos[0]
			tag = pos[1][0:2]

			if tag in POS:
				word = wordnet_lemmatizer.lemmatize(word)
				if word in lexicon_keys:
					found += 1
					values = LEXICON[word]
					if values['Anger'] == 1:
						emotion["anger_prob"] += 1
					elif values['Disgust'] == 1:
						emotion["disgust_prob"] += 1
					elif values['Fear'] == 1:
						emotion["fear_prob"] += 1
					elif values['Joy'] == 1:
						emotion["happy_prob"] += 1
					elif values['Sadness'] == 1:
						emotion["sad_prob"] += 1
					elif values['Surprise'] == 1:
						emotion["surprise_prob"] += 1
				else:
					not_found += 1
					emotion["emotionless_prob"] += 1

		t = sum(emotion.values())
		if t > 0:
			for emo, val in emotion.iteritems():
				FEATURES[i][emo] = val * 1.0 / t


		# Adding previous features
		if i > 1:
			FEATURES[i]["prev1_emotion"] = EMOTIONS[i-1]
		if i > 2:
			FEATURES[i]["prev2_emotion"] = EMOTIONS[i-2]
		if i > 3:
			FEATURES[i]["prev3_emotion"] = EMOTIONS[i-3]

	k += 1
	
	if line.strip() == "":
		k = 0

print found
print not_found

with open(other_features_dict['Friends'], 'w') as outfile:
	json.dump(FEATURES, outfile)
# print pp.pprint(FEATURES)