import numpy as np

#OneHot2Base = { 100000010: 'A', 00101000: 'U', 01000010: 'G', 00100100: 'C'}
#OneHot2Dot = { '(': 1000000, ')': 0000001, '.': 0001000}
#Base2OneHot = { 'A': [1 0 0 0 0 0 0 1 0], 'U': [0 0 1 0 1 0 0 0], 'G': [0 1 0 0 0 0 1 0], 'C': [ 0 0 1 0 0 1 0 0]}
#Dot2OneHot = { '(': 1000000, ')': 0000001, '.': 0001000}

def Base2OneHot(file):
    sequences = []
    with open(file) as f:
        for line in f:
            seq = []
            line = line.rstrip('\n')
            for char in line:
                if str(char) == 'A':
                    seq.append([1, 0, 0, 0, 0, 0, 1, 0])
                elif str(char) == 'U':
                    seq.append([0, 0, 1, 0, 1, 0, 0, 0])
                elif str(char) == 'G':
                    seq.append([0, 1, 0, 0, 0, 0, 1, 0])
                else:
                    seq.append([0, 0, 1, 0, 0, 1, 0, 0])
            sequences.append(np.array(seq))
    return np.array(sequences)

def Dot2OneHot(file):
    sequences = []
    with open(file) as f:
        for line in f:
            seq = []
            line = line.rstrip('\n')
            for char in line:
                if str(char) == '(':
                    seq.append([1, 0, 0, 0, 0, 0, 0])
                elif str(char) == ')':
                    seq.append([0, 0, 0, 0, 0, 0, 1])
                else:
                    seq.append([0, 0, 0, 1, 0, 0, 0])
            sequences.append(np.array(seq))
    return np.array(sequences)


filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
filename2 = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'
sequences = Base2OneHot(filename)
print(sequences[0])
dots = Dot2OneHot(filename2)
print(dots[0])
