import numpy as np
import glob, sys, os
import time

dir = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/seed_structures/'
new_dir = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/doNotTouch/'

def str_append(s):
    output = ''
    output += s
    return output

def readFile(file):
    record = {}
    dots = ''
    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            l = [x for x in line.split(' ') if x != '']
            n = int(l[0])
            base = l[1]
            partner = int(l[4])
            record[n] = [base, partner]
    seq = "".join([record[x][0] for x in sorted(record.keys())])
    for x in sorted(record.keys()):
        val = record[x][1]
        if val > x:
            s = '('
            dots += str_append(s)
        elif val == 0:
            s = '.'
            dots += str_append(s)
        elif val < x:
            s = ')'
            dots += str_append(s)
    return seq, dots


seq_files = os.listdir('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/seed_structures/')

# There are approximately 44000 sequences in our dataset. We wish to isolate 5% of our data to use for pure testing.
# Therefore we will select 2200 sequences to remove from our dataset.


condition = 2200
iter = 0
for file in seq_files:
    name = file
    if iter < condition:
        x = np.random.random()
        if x < 0.3:
            os.rename(dir + file, new_dir + file)
            iter += 1
print('Removed %d files from seed_structures to doNotTouch!' %(len(doNotTouch)))

sequences = []
strandLengths = []
dotBrackets= []
bonds = []
