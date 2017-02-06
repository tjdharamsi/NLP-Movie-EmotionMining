episode = "wd"

original_srt_file = episode + ".srt"
emotion_srt_file = episode + "_Emotion.srt"
emotion_file = episode + ".emotion"

fr = open(original_srt_file, "r")

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




# Length of each clip
time = 1

def total_seconds(time):
	hour, minute, sec = time.split(":")
	return int(hour) * 3600 + int(minute) * 60 + int(sec)

def time_difference(start, end):
	global time

	start_sec = total_seconds(start)
	end_sec = total_seconds(end)

	if end_sec - start_sec > time * 60:
		return True
	else:
		return False

length = 0

flag = 0

starting_time = 0

emotion = {"anger" : 0, "disgust":0, "emotionless":0, "fear":0, "happy":0, "sad":0, "surprise":0}
timeline = []
for line in fr:
	if k == 0:
		emotion[EMOTIONS[i]] += 1

	elif k == 1:
		line = line.strip()
		start, end = line.split("-->")
		start = start[0:8]
		end = end[1:9]

		if flag == 0:
			flag = 1
			starting_time = start
			
		if time_difference(starting_time, end):
			flag = 1
			# Dump the emotion array
			timeline.append(emotion)

			starting_time = start
			emotion = {"anger" : 0, "disgust":0, "emotionless":0, "fear":0, "happy":0, "sad":0, "surprise":0}


	k += 1

	if line.strip() == "":
		k = 0
		i += 1	

timer = 0

fw = open("emotion_group_wd.csv", "w")
for emo in timeline:
	for k,v in emo.iteritems():
		fw.write(str(k) + ", " + str(timer) + ", " + str(v) + "\n")
	timer += time