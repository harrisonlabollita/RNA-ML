import numpy as np
import matplotlib.pyplot as plt

# Enumerating RNA Structures

def can_pair(a, b):
    if a == 'A' and b == 'U':
        return True
    elif a == 'U' and b == 'A':
        return True
    elif a == 'G' and b == 'C':
        return True
    elif a == 'C' and b == 'G':
        return True
    elif a == 'G' and b == 'U':
        return True
    elif a == 'U' and b == 'G':
        return True
    else:
        return False

def possible_pairs(sequence):
    # Create a matrix of all possible pairs
    N = len(sequence)
    B = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if can_pair(sequence[i], sequence[j]):
                B[i, j] = 1
                B[j, i] = 1
    return B


def possible_stems(pair_matrix):
    N = len(pair_matrix)

    PossibleStems = []

    BasePairsChecked = []

    minStemLength = 2
    minHairPin = 3
    maxI = N- 2*minStemLength - minHairPin+ 1
    for i in range(N):
        minJ = i + 2*minStemLength + minHairPin - 2
        for j in range(N-1, minJ, -1):

            if pair_matrix[i, j] == 1:

                if [i, j] not in BasePairsChecked:

                    stemPairs = [[i, j]]

                    lenStem = 0
                    endStem = False
                    while not endStem:

                        lenStem += 1
                        new_i = i + lenStem
                        new_j = j - lenStem

                        if new_i > N-1 or \
                         new_j < 0 or new_j - new_i <= minHairPin or \
                         pair_matrix[new_i, new_j] == 0:

                            endStem = True
                            lenStem -= 1
                        else:
                            stemPairs.append([new_i, new_j])
                            BasePairsChecked.append([new_i, new_j])

                    if len(stemPairs) >= minStemLength:
                        PossibleStems.append(stemPairs)
    return PossibleStems

def mutualStems(stem1, stem2):
    for i in range(len(stem1)):
        for j in range(len(stem2)):

            if stem1[i][0] == stem2[j][0] or \
             stem1[i][0] == stem2[j][1] or \
             stem1[i][1] == stem2[j][0] or \
             stem1[i][1] == stem2[j][1]:
                return False
    return True

def compatiblilityMatrix(possibleStems):
    N_stems = len(possibleStems)
    compatibility_matrix = np.zeros((N_stems, N_stems))
    for i in range(N_stems):
        for j in range(N_stems):
            if i == j:
               compatibility_matrix[i, j] = 1
            else:
                stem1 = possibleStems[i]
                stem2 = possibleStems[j]

                if mutualStems(stem1, stem2):
                    compatibility_matrix[i, j] = 1
                    compatibility_matrix[j, i] = 1
    return compatibility_matrix

def isCompatible(array_of_stems, next_stem):
    for i in range(len(array_of_stems)):
        if mutualStems(array_of_stems[i], next_stem):
            pass
        else:
            return False
    return True


def compatibleStems(matrix, stems):
    N_stems = len(stems)
    compat_Stems= []
    for i in range(N_stems):
        for j in range(N_stems):
            if i != j and matrix[i, j] == 1:
                if len(compat_Stems) == 0:
                    s_j = j
                    compat_Stems.append(stems[i])
                    compat_Stems.append(stems[j])
                else:
                    for k in range(j, N_stems):
                        if isCompatible(compat_Stems, stems[k]) and s_j > k:
                            s_j = k
                            compat_Stems.append(stems[k])
    return compat_Stems

def calculateTransitionRates():

    return rate

################################# EXAMPLE ######################################
#                                                                              #
#                                        8                                     #
# 0   C     (      16                    A                                     #
# 1   G     (      15                  /   \                                   #
# 2   G     (      14               7 A     A  9                               #
# 3   U     (      13               6  G - U  10                               #
# 4   C     (      12               5  G - C  11                               #
# 5   G     (      11               4  C - G  12                               #
# 6   G     (      10               3  U - A  13                               #
# 7   A     .      -1               2  G - U  14                               #
# 8   A     .      -1               1  G - C  15                               #
# 9   C     .      -1               0  C - G  16                               #
# 10  U     )       6                      G  17                               #
# 11  C     )       5                      U  18                               #
# 12  G     )       4                      U  19                               #
# 13  A     )       3                      G  20                               #
# 14  U     )       2                      A  21                               #
# 15  C     )       1                      A  22                               #
# 16  G     )       0                      C  23                               #
# 17  G     .      -1                      U  24                               #
# 18  U     .      -1                      C  25                               #
# 19  U     .      -1                      U  26                               #
# 20  G     .      -1                      A  27                               #
# 21  A     .      -1                      U  28                               #
# 22  A     .      -1                      C  29                               #
# 23  C     .      -1                                                          #
# 24  U     .      -1                                                          #
# 25  C     .      -1                                                          #
# 26  U     .      -1                                                          #
# 27  A     .      -1                                                          #
# 28  U     .      -1                                                          #
# 29  C     .      -1                                                          #
#                                                                              #
################################################################################

seq = 'CGGUCGGAACUCGAUCGGUUGAACUCUAUC'
mat = possible_pairs(seq)
stems = possible_stems(mat)
compat = compatiblilityMatrix(stems)
poss_stems = compatibleStems(compat, stems)
for i in range(len(stems)):
    print(stems[i])
