# stress_test.py
# AUTHOR: Harrison LaBollita 
# DATE: 1.22.20
# 
import glob, sys, os
import gillespie as gill
import numpy as np
import time
def readFile(file):
    f = open(file, 'r')
    for i, line in enumerate(f):
        if i == 1:
            seq = line.rstrip()
        if i == 2:
            dots = line.split(' ')[1]
            dots.rstrip()
    return seq, dots

sequences = '/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/src.txt'

#seq_files = glob.glob('/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/bad_seq/*')
print('Generating output file')
outputFile = open('str_test.txt', 'w+')
with open(sequences, 'r') as file:
    for line in file:
        seq = line.rstrip()
        seq_length = len(seq)
        print('Starting...%s' %(seq))
        start = time.time()
        G = gill.Gillespie(seq, [], maxTime = 5, toPrint = False, initTime = True)
        structure =G.runGillespie()
        initialTime = G.initializationTime
        stop = time.time()
        print('Finished!')
        outputFile.write('%d  %0.2f %0.2f\n' %(seq_length, (stop - start), initialTime))

#for file in seq_files:
#    seq, dots = readFile(file)
#    seqLength = len(seq)
#    if seqLength <= 130:
#        print('Starting....%s' %(file))
#        start = time.time()
#        G = gill.Gillespie(seq, [], maxTime =5, toPrint = False)
#        structure = G.runGillespie()
#        stop = time.time()
#        print('Finished!')
#        outputFile.write('%d %0.2f\n' %(seqLength, (stop -start)))

