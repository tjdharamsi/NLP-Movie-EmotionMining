episode = "combined"

original_srt_file = episode + ".srt"
fr = open(original_srt_file, "r")
fw = open("new_combined.srt", "w")

i = 1	
k = 0
for line in fr:
	if k == 0:
		fw.write(str(i) + "\n")

	else:
		fw.write(line)

	k += 1

	if line.strip() == "":
		k = 0
		i += 1	

