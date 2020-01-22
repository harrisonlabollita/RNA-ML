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

seq_files = glob.glob('/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/bad_seq/*')

outputFile = open('str_test.txt', 'w+')
for file in seq_files:
    seq, dot = readFile(file)
    seq_length = len(seq)
    start = time.time()
    G = gill.Gillespie(seq, [], maxTime = 5, toPrint = False)
    structure =G.runGillespie()
    stop = time.time()
    outputFile.write('%d  %0.2f' %(seq_length, (stop - start)))

