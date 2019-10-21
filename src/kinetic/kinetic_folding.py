import numpy as np
import matplotlib.pyplot as plt

# Enumerating RNA Structures

def can_pair(A, B):
    if A == 'A' and B == 'U' or A == 'U' and B == 'A':
        return True
    elif A == 'G' and B == 'C' or A == 'C' and B == 'G':
        return True
    elif A == 'G' and B == 'C' or A == 'C' and B == 'G':
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
                B[i][j] = 1
    return B


def possible_stems(pair_matrix):
    N = len(pair_matrix)

    PossibleStems = []

    BasePairsChecked = []

    minStemLength = 2
    minHairPin = 3
    numStems = 0

    for i in range(N):

        for j in range(N-1, i, -1):

            if pair_matrix[i, j] == 1:

                if [i, j] not in BasePairsChecked:

                    stemPairs = [[i, j]]

                    lenStem = 0
                    endStem = False

                    while not endStem:

                        lenStem += 1
                        new_i = i + lenStem
                        new_j = j - lenStem

                        if new_i > N or new_j < 0 or abs(new_j - new_i) <= minHairPin or pair_matrix[new_i, new_j] == 0:
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
                    compatibility_matrix[i,j] = 1

    return compatibility_matrix


seq = 'UUCGCAGCGAAUUCUACAGUCAAUAUGGUG'

mat = possible_pairs(seq)
stems = possible_stems(mat)
compat = compatiblilityMatrix(stems)

def isCompatible(array_of_stems, next_stem):
    for i in range(len(array_of_stems)):
        if mutualStems(array_of_stems[i], next_stem):
            pass
        else:
            return False
    return True


def compatibileStems(matrix, stems):
    N_stems = len(stems)
    compatibileStems = []
    compatibleStems.append(stems[0])
    for i in range(N_stems):
        for j in range(N_stems):
            if i != j and isCompatible(compatibleStems, ):
                compatibleStems.append(compatibleStems, )
