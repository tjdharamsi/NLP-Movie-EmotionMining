episode = "Titanic"

original_srt_file = episode + ".srt"
emotion_srt_file = episode + "_Emotion.srt"
emotion_file = episode + ".emotion"

fr = open(original_srt_file, "r")
fw = open(emotion_srt_file, "w")

EMOTIONS = {}
i = 1
# Read emotion file
f = open(emotion_file)

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

