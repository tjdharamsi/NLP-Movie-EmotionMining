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
	if k > 1:
		if k == 2:
			fw.write(EMOTIONS[i])
			fw.write("\n")
	else:
		fw.write(line)

	k += 1

	if line.strip() == "":
		fw.write("\n")
		k = 0
		i += 1	

