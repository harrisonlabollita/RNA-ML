import numpy as np
import sys
import glob


def readFile(file):
    record = {}
    dot = ''
    with open(file, 'r') as f:
        for line in f.readlines()[1:]:
            l = [x for x in line.split() if x != '']
            n = int(l[0])
            base = l[1]
            partner = int(l[4])
            record[n] = [base, partner]
    seq = "".join([record[x][0] for x in sorted(record.keys())])
    return seq, record

def isPsuedoknotted(structure):
    for i in range(len(structure):
        for j in range(len(structure):

seq_files = glob.glob('/Users/harrisonlabollit/Library/Mobile Documents/com~apple_CloudDocs/Arizona State University/Sulc group/data_set/ct_files/*')


for file in seq_files:
    sequence, structure = readFile(file) 

