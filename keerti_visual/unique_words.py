import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

f = open("wd.srt")

k = 0
POS = ['NN','VB','JJ','RB']

words = {}

i = 0
for line in f:
	if k > 1:
		# Finding POS percents
		line = line.decode("utf-8").strip()
		tokens = nltk.word_tokenize(line)
		pos_tag = nltk.pos_tag(tokens)

		for pos in pos_tag:
			tag = pos[1][0:2]
			# print tag
			if tag in POS:
				word = pos[0].lower()
				word = wordnet_lemmatizer.lemmatize(word)

				if word not in words.keys():
					words[word] = 1
				else:
					words[word] += 1

	k += 1
	
	if line.strip() == "":
		k = 0

import operator
sorted_x = sorted(words.items(), key=operator.itemgetter(1))
print sorted_x

f=open("w_results.txt",'w')
for i in sorted_x:
	f.write(str(i)+"\n")
f.close()