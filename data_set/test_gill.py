#import gillespie as GILLESPIE
import glob, sys, os
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/kinetic_RNA/')
import gillespie as GILLESPIE
import numpy as np
import matplotlib.pyplot as plt

def str_append(s):
    output = ''
    output += s
    return output

def readFile(file):
    f = open(file, 'r')
    for i, line in enumerate(f):
        if i == 1:
            seq = line.rstrip()
        if i == 2:
            dots = line.split(' ')[1]
            dots.rstrip()
    return seq, dots

def compareStructs(pred, answer):
    count = 0
    for i in range(len(pred)):
        if pred[i] != answer[i]:
            count +=1
    return count



seq_files = glob.glob('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/data_set/bad_seq/*')

total_misses = []
for file in seq_files:
    seq, dot = readFile(file)
    G = GILLESPIE.Gillespie(seq, [], maxTime = 5, toPrint = False)
    structure = G.runGillespie()
    mistakes = compareStructs(structure, dot)
    mistakes /= len(seq)
    total_misses.append(mistakes)

print('Accuracy:' (total_misses.count(0))/len(total_misses))
print('Mean:', np.mean(total_misses))
print('Max:', np.max(total_misses))
print('Min:', np.max(total_misses))
