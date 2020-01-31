import numpy as np
import os, sys, glob
import matplotlib.pyplot as plt
sys.path.append('/Users/harrisonlabollita/Arizona State University/Sulc group/src/kinetic_RNA/')
import gillespie as GILLESPIE
import time


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
        if i == 3:
            rnafold = line.split(' ')[1]
            rnafold.rstrip()
    return seq, dots, rnafold

def compareStructs(pred, answer):
    count = 0
    for i in range(len(pred)):
        if pred[i] != answer[i]:
            count +=1
    return count

def switchRep(structure):
    structure.replace('[', '(')
    structure.replace(']', ')')
    return structure

def getData(length, files):
    sequences = []
    brackets = []
    viennaRNA = []
    for file in files:
        seq, dot, rnafold = readFile(file)
        if len(seq) <= length:
            sequences.append(seq)
            brackets.append(dot)
            viennaRNA.append(rnafold)
    return sequences, brackets, viennaRNA

seq_files = glob.glob('/Users/harrisonlabollita/Arizona State University/Sulc group/data_set/bad_seq/*')
sequences, dotbrackets, viennaRNA = getData(75, seq_files)


total_misses = []
rna_misses = []
for i in range(len(sequences)):

    seq = sequences[i]
    dot = dotbrackets[i]
    rnafold = viennaRNA[i]

    start = time.time()
    print('Starting...{}/{}, Length: {}'.format(i+1, len(sequences), len(seq)))
    G = GILLESPIE.Gillespie(seq, [], maxTime = 5, toPrint = False, initTime = False)
    structure = switchRep(G.runGillespie())
    mistakes = compareStructs(structure, dot)
    rna_mistake = compareStructs(rnafold, dot)
    rna_mistake /= len(seq)
    mistakes /= len(seq)
    total_misses.append(mistakes)
    rna_misses.append(rna_mistake)
    stop = time.time()
    print('Finishing...{}/{}, Time: {:0.4f}, myCorrect(%): {:0.2f} RNAfoldCorrect(%): {:0.2f}'.format(i+1, len(sequences), abs(stop-start), 1-mistakes, 1-rna_mistake))

print('myMean:', np.mean(1- total_misses))
print('myMax:', np.max(1- total_misses))
print('myMin:', np.max(1- total_misses))
print('rnaMean:', np.mean(1- rna_misses))
print('rnaMax:', np.max(1- rna_misses))
print('rnaMin:', np.max(1- rna_misses))
