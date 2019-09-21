import numpy as np
import sys
import csv

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNA_data_set.csv'

def readData(filename):
    #
    # Input: filename (as .csv)
    # Output: RNA sequences, dot Bracket representation, and free energy
    with open(filename) as f:
        next(f)
        sequs = csv.reader(f)
        data = []
        for seq in sequs:
            data.append(seq)
        data = np.asarray(data)
        seq, dot, free = refineData(data)
    return seq, dot, free

def refineData(data):
    freeEnergy = []
    sequences = []
    dotBrackets = []
    for i in range(len(data)):
        sequences.append(data[i][0])
        dotBrackets.append(data[i][1])
        freeEnergy.append(data[i][2])
    sequences = np.asarray(sequences)
    dotBrackets = np.asarray(dotBrackets)
    freeEnergy = np.asarray(freeEnergy)
    return sequences, dotBrackets, freeEnergy


def findBonds(brackets):
    bonds = []
    for i in range(len(brackets)):
        count = 0
        for j in range(len(brackets[i])):
            if brackets[i][j] == '(' or brackets[i][j] == ')':
                count +=1
            else:
                count +=0
        count /= 2
        bonds.append(count)
    return bonds


sequences, dotBrackets, freeEnergy = readData(filename)


lengths = []
N = len(sequences)
for i in range(N):
    lengths.append( len(sequences[i]) )

bonds = findBonds(dotBrackets)


print('+++++++++DATA STATISTICS++++++++++++')
print('Length of dataset: %d' %(len(lengths)))
print('Shortest RNA sequence: %d' %(min(lengths)))
print('Longest RNA sequence: %d' %(max(lengths)))
print('Mean: %0.2f' % (np.mean(lengths)))
print('StDev: %0.2f' %(np.std(lengths)))
print('+++++++++++++++++++++++++++++++++++++')
print('Maximum bonds: %d' %(max(bonds)))
print('Minimum bonds: %d' %(min(bonds)))
print('Mean: %0.2f' %(np.mean(bonds)))
print('StDev: %0.2f' %(np.std(bonds)))
