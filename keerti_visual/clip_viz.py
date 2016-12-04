fr = open("Titanic.srt", "r")
fw = open("Titanic_Emotion.srt", "w")

EMOTIONS = {}
i = 1
# Read emotion file
f = open("Titanic.emotion")
for line in f:
	line = line.strip()
	EMOTIONS[i] = line
	i += 1

k = 0
i = 1

for line in fr:
	print i
	print line.strip()

	if k > 1:
		fw.write(EMOTIONS[i])

	else:
		fw.write(line)

	k += 1
	if line.strip() == "":
		k = 0
		i += 1	

