import csv
import os
total = 0
problem_set = {
    '怪怪' : [],
    '沒念出來' : [],
}

with open(f'./phph.csv', 'r', encoding='utf8') as fd:
    reader = csv.reader(fd)
    for row in reader:
        tw, tw_p = row
        if tw:
            total += 1
        if tw_p:
            try:
                problem_set[tw_p].append(tw)
            except:
                continue

for each in problem_set:
    for p in problem_set[each]:
        os.system(f'cp gen_audio/56/taiwanese/{p} gen_audio/56/')