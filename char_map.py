import os
char_map = {}
index_map = {}
for line in open("tongji_dict"):
    line = line.strip("\n")
    ch, index = line.split(" ")
    ch = ch.decode('utf8')
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[408] = ' '
print len(char_map.keys()), len(index_map.keys())
