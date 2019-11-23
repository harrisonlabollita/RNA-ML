import numpy as np
import copy
import scipy
from scipy import sparse as sp

############################## HELPER FUNCTIONS ################################
# Functions taken from RFE Landscape and used independuntly from the calculate
# FE landscapre funciton in the RFE class. Paper can be
# found at (https://www.biorxiv.org/content/10.1101/338921v1).
# Most of the comments and descriptions from functions have been removed to increase
# readability.

def twoListsShareElement(a,b):
# =============================================================================
#     Check if two lists, a and b, share any items. This is supposedly the fastest way to test this.
#    https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
#    Returns True if the lists share any elements, False otherwise
# =============================================================================
    return(not set(a).isdisjoint(b))

def bpsList2structure(bpsList):
# =============================================================================
#     Given a list of base pairs, make them in "structure" format.
# =============================================================================
    bpsList = sorted(bpsList)
    structure = []
    startOfStem = False
    stem1 = [] #first nts of stem
    stem2 = [] #complementary nts of stem
    for i in range(len(bpsList)): #for each base pair
        firstBP = bpsList[i][0]
        secondBP = bpsList[i][1]

        if i: #i.e. if it's not the first base pair being considered
            #check if it's a continuation of the previous stem.
            if firstBP - stem1[-1] == 1:
                if secondBP - stem2[-1] == 1: #we're dealing with a parallel stem.
                    startOfStem = False
                elif secondBP - stem2[-1] == -1: #antiparallel (normal) stem
                    startOfStem = False
                else:
                    startOfStem = True
            else:
                startOfStem = True

        if not startOfStem:
            stem1.append(firstBP)
            stem2.append(secondBP)
        else:
            structure.append(stem1 + stem2)
            stem1 = [firstBP]
            stem2 = [secondBP]

        #if we've reached the end, make sure to put in the last base pair
        if i == len(bpsList) - 1:
            structure.append(stem1 + stem2)
    return(structure)
def bpsList2structBPs(bpsList):
# =============================================================================
#     Given a list of base pairs, put them in structBPs format, meaning take the list of base pairs
#     and make a list of stems, where each stem is itself a list of base pairs in that stem.
#     For exmample, a bpList would be [[1,10],[2,9],[3,8],[15,25],[16,24],[18,22],[17,23]]
#     The corresponding structBPs would be [[[1,10],[2,9],[3,8]],[[15,25],[16,24],[17,23],[18,22]]]
#     and the corresponding structure would be [[1,2,3,10,9,8],[15,16,17,18,25,24,23,22]]
# =============================================================================

    bpsList = sorted(bpsList)
    structBPs = [[]]
    startOfStem = False
    for i in range(len(bpsList)): #for each base pair
        firstBP = bpsList[i][0]
        secondBP = bpsList[i][1]

        if i: #i.e. if it's not the first base pair being considered
            #check if it's a continuation of the previous stem.
            if firstBP - structBPs[-1][-1][0] == 1:
                if secondBP - structBPs[-1][-1][1] == 1: #we're dealing with a parallel stem.
                    startOfStem = False
                elif secondBP - structBPs[-1][-1][1] == -1: #antiparallel (normal) stem
                    startOfStem = False
                else:
                    startOfStem = True
            else:
                startOfStem = True

        if not startOfStem:
            structBPs[-1].append([firstBP,secondBP])
        else:
            structBPs.append([[firstBP,secondBP]])
    return(structBPs)

def frozenStemsFromFrozenBPs(frozenBPs, STableBPs, numStems): #still need to check more carefully
    if not frozenBPs:
        return([])

        #if two or more base pairs of frozenBPs are adjacent, then any stem that includes them
        #must include all of them to be considered (otherwise there'd be no way to include the others).
    frozenStructBPs = bpsList2structBPs(frozenBPs) #a list of frozen stems where each stem is in base-pair (bp) format
    frozenStems = [[i for i in range(numStems) if all([k in STableBPs[i] for k in j])] for j in frozenStructBPs]
    frozenStems = [i for i in frozenStems if i] #remove empty lists
    return(frozenStems)

def bondFreeEnergiesRNARNA():

#    by (4*second index) + third index
    bondEnergyMatrixRNARNA = np.array([[-3.9, 2, -3.5, -6.82, -2.3, 6, -11.4, -0.3, -3.1, -10.48, -3.5, -3.21, -9.38, 4.6, -8.81, -1.7],
                                       [-9.1, -5.6, -5.6, -10.44, -5.7, -3.4, -13.39, -2.7, -8.2, -10.64, -9.2, -5.61, -10.48, -5.3, -12.11, -8.6],
                                       [-5.2, -4, -5.6, -12.44, -7.2, 0.5, -14.88, -4.2, -7.1, -13.39, -6.2, -8.33, -11.4, -0.3, -12.59, -5],
                                       [-4, -6.3, -8.9, -7.69, -4.3, -5.1, -12.44, -1.8, -3.8, -10.44, -8.9, -6.99, -6.82, -1.4, -12.83, 1.4],
                                       [3.4, 2, -3.5, -12.83, -2.3, 6, -12.59, -0.3, 0.6, -12.11, -3.5, -13.47, -8.81, 4.6, -14.59, -1.7],
                                       [-4.8, -6.3, -8.9, -6.99, -4.3, -5.1, -8.33, -1.8, -3.1, -5.61, 1.5, -9.26, -3.21, -1.4, -13.47, 1.4]])
    #matrix of enthalpies (deltaH). Units are kcal/mol.

    bondEntropyMatrixRNARNA = np.array([[-10.2, 9.6, -8.7, -19, -5.3, 21.6, -29.5, 1.5, -7.3, -27.1, -8.7, -8.6, -26.7, 17.4, -24, -2.7],
                                        [-24.5, -13.5, -13.4, -26.9, -15.2, -7.6, -32.7, -6.3, -21.8, -26.7, -24.6, -13.5, -27.1, -12.6, -32.2, -23.9],
                                        [-13.2, -8.2, -13.9, -32.5, -19.6, 3.9, -36.9, -12.2, -17.8, -32.7, -15.1, -21.9, -29.5, -2.1, -32.5, -14],
                                        [-9.7, -17.1, -25.2, -20.5, -11.6, -14.6, -32.5, -4.2, -8.5, -26.9, -25, -19.3, -19, -2.5, -37.3, 6],
                                        [10, 9.6, -8.7, -37.3, -5.3, 21.6, -32.5, 1.5, 0, -32.2, -8.7, -44.9, -24, 17.4, -51.2, -2.7],
                                        [-12.1, -17.7, -25.2, -19.3, -11.6, -14.6, -21.9, -4.2, -11.2, -13.5, 2.1, -30.8, -8.6, -2.5, -44.9, 6]])
    #matrix of entropies (deltaS). Units are initially eu, but then converted to kcal/(mol*K).

    bondEntropyMatrixRNARNA /= 1000 #to convert from eu (entropy units) to kcal/(mol*K)

    return(bondEnergyMatrixRNARNA, bondEntropyMatrixRNARNA)

def bondFreeEnergiesDNADNA():


    bondFreeEnergyMatrixDNADNA = np.array([[0.61, 0.88, 0.14, -1.0, 0.77, 1.33, -1.44, 0.64, 0.02, -1.28, -0.13, 0.71, -0.88, 0.73, 0.07, 0.69],
                                           [0.43, 0.75, 0.03, -1.45, 0.79, 0.7, -1.84, 0.62, 0.11, -2.17, -0.11, -0.47, -1.28, 0.4, -0.32, -0.21],
                                           [0.17, 0.81, -0.25, -1.3, 0.47, 0.79, -2.24, 0.62, -0.52, -1.84, -1.11, 0.08, -1.44, 0.98, -0.59, 0.45],
                                           [0.69, 0.92, 0.42, -0.58, 1.33, 1.05, -1.3, 0.97, 0.74, -1.45, 0.44, 0.43, -1.0, 0.75, 0.34, 0.68]])
    #matrix of enthalpies (deltaH). Units are kcal/mol.

    bondEnergyMatrixDNADNA = np.array([[1.2, 2.3, -0.6, -7.9, 5.3, 0.0, -8.4, 0.7, -0.7, -7.8, -3.1, 1.0, -7.2, -1.2, -2.5, -2.7],
                                       [-0.9, 1.9, -0.7, -8.5, 0.6, -1.5, -8, -0.8, -4, -10.6, -4.9, -4.1, -7.8, -1.5, -2.8, -5],
                                       [-2.9, 5.2, -0.6, -8.2, -0.7, 3.6, -9.8, 2.3, 0.5, -8, -6, 3.3, -8.4, 5.2, -4.4, -2.2],
                                       [4.7, 3.4, 0.7, -7.2, 7.6, 6.1, -8.2, 1.2, 3, -8.5, 1.6, -0.1, -7.9, 1.0, -1.3, 0.2]])
    #matrix of free energies (deltaG). Units are kcal/mol.

    bondEntropyMatrixDNADNA = -(bondFreeEnergyMatrixDNADNA - bondEnergyMatrixDNADNA) / (273.15+37)
    #matrix of entropies (deltaS). Units are converted to kcal/(mol*K) by dividing by 310.15 since the free energies were measured at 310.15 K

    return(bondEnergyMatrixDNADNA, bondEntropyMatrixDNADNA)

def bondFreeEnergiesRNADNA():

    #100 is put in place of elements for which there aren't published parameters
    bondFreeEnergyMatrixRNADNA = np.array([[1.07, 100, 100, -1.0, 100, 1.64, -2.1, 100, 100, -1.8, 0.31, 100, -0.9, 100, 100, 0.63],
                                           [0.90, 100, 100, -0.9, 100, 1.04, -2.1, 100, 100, -1.7, 0.14, 100, -0.9, 100, 100, 0.49],
                                           [0.51, 100, 100, -1.3, 100, 0.96, -2.7, 100, 100, -2.9, -0.58, 100, -1.1, 100, 100, 0.18],
                                           [1.13, 100, 100, -0.6, 100, 1.15, -1.5, 100, 100, -1.6, 0.44, 100, -0.2, 100, 100, 1.07],
                                           [1.36, 100, 100, -1.0, 100, 1.70, -0.9, 100, 100, -1.3, 0.50, 100, -0.6, 100, 100, 1.21],
                                           [0.19, 100, 100, -2.1, 100, 0.73, -2.1, 100, 100, -2.7, -0.83, 100, -1.5, 100, 100, -0.02],
                                           [0.21, 100, 100, -1.8, 100, 0.46, -1.7, 100, 100, -2.9, -0.33, 100, -1.6, 100, 100, 0.14],
                                           [1.85, 100, 100, -0.9, 100, 1.88, -0.9, 100, 100, -1.1, 0.97, 100, -0.2, 100, 100, 1.03]])
    #matrix of stacking free energies (deltaG). Units are kcal/mol.

    bondEnergyMatrixRNADNA = np.array([[-4.3, 100, 100, -7.8, 100, -8.8, -5.9, 100, 100, -9.1, -3.3, 100, -8.3, 100, 100, 0.6],
                                       [5.5, 100, 100, -9.0, 100, 10.5, -9.3, 100, 100, -16.3, -8.9, 100, -7.0, 100, 100, -0.4],
                                       [-1.9, 100, 100, -5.5, 100, -0.1, -8.0, 100, 100, -12.8, -8.0, 100, -7.8, 100, 100, -11.6],
                                       [-1.7, 100, 100, -7.8, 100, -3.3, -8.6, 100, 100, -10.4, -5.8, 100, -11.5, 100, 100, -2.2],
                                       [3.0, 100, 100, -7.8, 100, -0.3, -9.0, 100, 100, -5.5, 1.1, 100, -7.8, 100, 100, -3.3],
                                       [-6.0, 100, 100, -5.9, 100, 9.3, -9.3, 100, 100, -8.0, -7.0, 100, -8.6, 100, 100, 0.1],
                                       [-10.5, 100, 100, -9.1, 100, -11.5, -16.3, 100, 100, -12.8, -16.5, 100, -10.4, 100, 100, -13.4],
                                       [5.6, 100, 100, -8.3, 100, 0.8, -7.0, 100, 100, -7.8, -3.7, 100, -11.5, 100, 100, 3.0]])
    #matrix of enthalpies (deltaH). Units are kcal/mol.

    bondEntropyMatrixRNADNA = -(bondFreeEnergyMatrixRNADNA - bondEnergyMatrixRNADNA) / (273.15+37)

    #for elements for which there aren't published parameters, average the values for RNARNA and DNADNA bonds.
    bondEnergyMatrixDNADNA, bondEntropyMatrixDNADNA = bondFreeEnergiesDNADNA()
    bondEnergyMatrixRNARNA, bondEntropyMatrixRNARNA = bondFreeEnergiesRNARNA()

    for i in range(8):
        for j in range(4):
            for k in range(4):
                if bondEntropyMatrixRNADNA[i,k] == 0:
                    if i < 4:
                        #bondFreeEnergyMatrixRNADNA[i,j,k] = (bondFreeEnergyMatrixDNADNA[i,j,k] + bondFreeEnergyMatrixRNARNA[i,j,k]) / 2;
                        bondEnergyMatrixRNADNA[i,4*j+k] = (bondEnergyMatrixDNADNA[i,4*j+k] + bondEnergyMatrixRNARNA[i,4*j+k]) / 2;
                        bondEntropyMatrixRNADNA[i,4*j+k] = (bondEntropyMatrixDNADNA[i,4*j+k] + bondEntropyMatrixRNARNA[i,4*j+k]) / 2;
                    else: #flip the stem upside down to put the X,Y pair second. The order is then TA, GC, CG, AU
                        #(i.e. the opposite order from the i<=4 case). Also, then need to switch j and k (i.e. switch X and Y).

                        #bondFreeEnergyMatrixRNADNA[i,4*j+k] = (bondFreeEnergyMatrixDNADNA[7-i,4*k+j] + bondFreeEnergyMatrixRNARNA[7-i,4*k+j]) / 2;
                        bondEnergyMatrixRNADNA[i,4*j+k] = (bondEnergyMatrixDNADNA[7-i,4*k+j] + bondEnergyMatrixRNARNA[7-i,4*k+j]) / 2;
                        bondEntropyMatrixRNADNA[i,4*j+k] = (bondEntropyMatrixDNADNA[7-i,4*k+j] + bondEntropyMatrixRNARNA[7-i,4*k+j]) / 2;

    return(bondEnergyMatrixRNADNA, bondEntropyMatrixRNADNA)

def bondFreeEnergies():
    bondEnergyMatrixDNADNA, bondEntropyMatrixDNADNA = bondFreeEnergiesDNADNA()
    bondEnergyMatrixRNARNA, bondEntropyMatrixRNARNA = bondFreeEnergiesRNARNA()
    bondEnergyMatrixRNADNA, bondEntropyMatrixRNADNA = bondFreeEnergiesRNADNA()
    terminal_AU_penalty_energy, terminal_AU_penalty_entropy, \
        terminal_AT_penalty_energy, terminal_AT_penalty_entropy = terminalATAUPenalties()
    unknownEnergyMatrix = np.zeros((1,1)); unknownEntropyMatrix = np.zeros((1,1))

    energyMatrices = [bondEnergyMatrixRNARNA, bondEnergyMatrixDNADNA, bondEnergyMatrixRNADNA,
                      np.array([[terminal_AU_penalty_energy, terminal_AT_penalty_energy]]), unknownEnergyMatrix]
    entropyMatrices = [bondEntropyMatrixRNARNA, bondEntropyMatrixDNADNA, bondEntropyMatrixRNADNA,
                       np.array([[terminal_AU_penalty_entropy, terminal_AT_penalty_entropy]]), unknownEntropyMatrix]

    return(energyMatrices, entropyMatrices)

def terminalATAUPenalties():
    terminal_AT_penalty_energy = 2.2; terminal_AT_penalty_entropy = (2.2-0.05)/310.15
    #from SantaLucia and Hicks review The Termodynamicso of DNA Structural Motifs

    terminal_AU_penalty_energy = 3.72; terminal_AU_penalty_entropy = (3.72 - 0.45)/310.15
    #from Xia et al. 1998

    return(terminal_AU_penalty_energy, terminal_AU_penalty_entropy,
             terminal_AT_penalty_energy, terminal_AT_penalty_entropy)

def freeEnergyMatrixIndices(sequenceInNumbers, firstNtIndex, firstBPIndex, secondNtIndex,
                            secondBPIndex, bound = [1,1], unmatchedBPPenalty = True):

    if not bound[0]: #if we accidentally have firstNt and firstBP not bound, flip the stem
        realSecondBPIndex = copy.copy(firstNtIndex); realFirstBPIndex = copy.copy(secondNtIndex)
        realSecondNtIndex = copy.copy(firstBPIndex); realFirstNtIndex = copy.copy(secondBPIndex)
        firstNtIndex = realFirstNtIndex; firstBPIndex = realFirstBPIndex
        secondNtIndex = realSecondNtIndex; secondBPIndex = realSecondBPIndex

    numNt = len(sequenceInNumbers)

    if (firstNtIndex < 0 or firstNtIndex > numNt - 1 or secondNtIndex < 0 or secondNtIndex > numNt - 1 or
            firstBPIndex < 0 or firstBPIndex > numNt - 1 or secondBPIndex < 0 or secondBPIndex > numNt - 1):
        firstNt = 0; firstBP = 0; secondNt = 0; secondBP = 0 #if anything is disallowed, return zero
    else:
        firstNt = sequenceInNumbers[firstNtIndex] #the identity of the nt (A is 1, C is 2, etc.)
        firstBP = sequenceInNumbers[firstBPIndex]
        secondNt = sequenceInNumbers[secondNtIndex]
        secondBP = sequenceInNumbers[secondBPIndex]

    if firstNt == 0 or firstBP == 0 or secondNt == 0 or secondBP == 0:
        return([0,0,0], 4) #if one of the sequence elements is unknown, return bpType = 4
#        This also means we don't need to worry about whether or not the base pairs sent
#       correspond to the linker or something.

#    if base pair could bind but aren't bound in this structure, Lu, Turner, Mathews (NAR, 2006) say
#    that we should treat them as an AC pair (where A replaces the purine and C the pyrimidine)
    if unmatchedBPPenalty and not bound[1]:
        if ((secondNt == 1 and secondBP == 4) or (secondNt == 4 and secondBP == 1) or
                (secondNt == 2 and secondBP == 3) or (secondNt == 3 and secondBP == 2) or
                (secondNt == 3 and secondBP == 4) or (secondNt == 4 and secondBP == 3)):
            if secondNt == 1 or secondNt == 3: #if secondNt is a purine
                secondNt = 1 #the purine is replaced with A
                secondBP = 2 #and the pyrimidine with C
            else:
                secondNt = 2
                secondBP = 1

# =============================================================================
#     Find type of bond
# =============================================================================
    if firstNt <=4 and firstBP <= 4 and secondNt <=4 and secondBP <=4:
        bpType = 0 #RNARNA

    elif firstNt > 4 and firstBP > 4 and secondNt > 4 and secondBP > 4:
        bpType = 1 #DNADNA

    elif firstNt <=4 and firstBP > 4 and secondNt <=4 and secondBP > 4:
        bpType = 2 #RNADNA

    elif firstNt > 4 and firstBP <=4 and secondNt > 4 and secondBP <=4:
        bpType = 2 #RNADNA
#       need to flip the stack upside down (i.e. just look at it as if the second strand were the first.)
        realSecondBP = copy.copy(firstNt); realFirstBP = copy.copy(secondNt)
        realSecondNt = copy.copy(firstBP); realFirstNt = copy.copy(secondBP)
        firstNt = realFirstNt; firstBP = realFirstBP; secondNt = realSecondNt; secondBP = realSecondBP


# =============================================================================
#     Get indices of matrix for specific pair of bps
# =============================================================================

    if bpType == 0: #we're dealing with an RNA/RNA bond
        #For the bondEnergyMatrix,
        #First index tells you if the first bp of the set is AU (0) CG (1) GC (2) UA (3) GU (4) or UG (5)
        #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or U(3)
        #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or U(3)
        if firstNt == 1 and firstBP == 4:
            basePair = 0
        elif firstNt == 2 and firstBP == 3:
            basePair = 1
        elif firstNt == 3 and firstBP == 2:
            basePair = 2
        elif firstNt == 4 and firstBP == 1:
            basePair = 3
        elif firstNt == 3 and firstBP == 4:
            basePair = 4
        elif firstNt == 4 and firstBP == 3:
            basePair = 5
        else:
            print('The deltaG function has a problem RNA/RNA')

        return([basePair, 4*(secondNt - 1) + (secondBP - 1)], bpType)
        #need -1 for RNA nts since nts are numbered 1-4 but indices are 0-3
#        Had to make the energy/entropy matrices 2D which is why secondNt and secondBP are
#        not being returned as separate indices, but as one index (4*secondNt + secondBP)

    elif bpType == 1: #we're dealing with a DNA/DNA pair

        if firstNt == 5 and firstBP == 8:
            basePair = 0
        elif firstNt == 6 and firstBP == 7:
            basePair = 1
        elif firstNt == 7 and firstBP == 6:
            basePair = 2
        elif firstNt == 8 and firstBP == 5:
            basePair = 3
        else:
            print('The deltaG function has a problem DNA/DNA')

        return([basePair, 4*(secondNt - 5) + (secondBP - 5)], bpType)
        #need -5 for DNA nts since nts are numbered 5-8 but indices are 0-3
#        Had to make the energy/entropy matrices 2D which is why secondNt and secondBP are
#        not being returned as separate indices, but as one index (4*secondNt + secondBP)

    elif bpType == 2: #we're dealing with an RNA/DNA bond


        if firstNt == 1 and firstBP == 8:
            basePair = 0
        elif firstNt == 2 and firstBP == 7:
            basePair = 1
        elif firstNt == 3 and firstBP == 6:
            basePair = 2
        elif firstNt == 4 and firstBP == 5:
            basePair = 3
        elif secondNt == 1 and secondBP == 8:
            basePair = 4
        elif secondNt == 2 and secondBP == 7:
            basePair = 5
        elif secondNt == 3 and secondBP == 6:
            basePair = 6
        elif secondNt == 4 and secondBP == 5:
            basePair = 7
        else:
            print('The deltaG function has a problem RNA/DNA')

        return([basePair,4*(secondNt - 1) + (secondBP - 5)], bpType)


    def checkCompatibility(C, C3, C4, numStems, linkedStems):


        considerC3andC4 = True
        minNumStemsInStructure = 2

        numSequences = 1
        onlyConsiderBondedStrands = False

        numStructures = 0
        structures = []

        prevNumStructures = -1 #keep track of this just so that we don't print the same statement multiple times

        for i in range(numStems):
            for j in range(i+1,numStems):
                if C[i,j]:
                    currStructure = [i,j]
                    lenCurrStructure = 2
                    k = j #the next stem we'll try adding (we're about to add one so that's why it's j and not j+1)
                    while lenCurrStructure >= 2:
                        while k < numStems - 1:
                            k += 1

                            mutuallyCompatible = True
                            #Check mutual 2-way compatibility between the stem we want to add and all stems
                            #present in the current structure.

                            for l in currStructure[:lenCurrStructure]: #the same as just "in currStructure" but written
                                #this way so that if we want to preallocate space it'll be easier
                                if not C[k,l]:
                                    mutuallyCompatible = False
                                    break

                            if considerC3andC4 and mutuallyCompatible:
#                                Check 3-way compatibility. Iterate over all pairs in current structure
                                for l,m in itertools.combinations(currStructure[:lenCurrStructure],2):
                                    if not C3[l,m,k]:
                                        mutuallyCompatible = False;
                                        break

#                                Check 4-way compatibility
                                if mutuallyCompatible and lenCurrStructure > 2:
                                    #don't actually need to specify and lenCurrStructure > 2 since itertools.combinations
#                                       would just return an empty set if lenCurrStructure == 2
#                                    Iterate over all triplets in current structure
                                    for l,m,n in itertools.combinations(currStructure[:lenCurrStructure],3):
                                        if not C4[l,m,n,k]:
                                            mutuallyCompatible = False;
                                            break
                            if mutuallyCompatible:
                                lenCurrStructure += 1
                                currStructure.append(k)

def calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers):

    unmatchedBPPenalty = True
    includeTerminalAUATPenalties = True
    considerAllAsTerminalMismatches = False
#        #Define energy (enthalpy; deltaH; units of kcal/mol) and entropy (deltaS; units of kcal/(mol*K)) matrices for bonds.
    energyMatrices, entropyMatrices = bondFreeEnergies()
    stemEnergies = np.zeros(numStems)
    stemEntropies = np.zeros(numStems)

    if includeTerminalAUATPenalties:
            (terminal_AU_penalty_energy, terminal_AU_penalty_entropy,
             terminal_AT_penalty_energy, terminal_AT_penalty_entropy) = terminalATAUPenalties()
    else:
            (terminal_AU_penalty_energy, terminal_AU_penalty_entropy,
             terminal_AT_penalty_energy, terminal_AT_penalty_entropy) = [0,0,0,0]


    RNARNACount = sp.lil_matrix((6,16),dtype=int) #np.zeros((6,4,4),dtype = int)
    DNADNACount = sp.lil_matrix((4,16),dtype=int) #np.zeros((4,16),dtype = int)
    RNADNACount = sp.lil_matrix((8,16),dtype=int) #np.zeros((8,16),dtype = int)
    terminalAUATCount = sp.lil_matrix((1,2),dtype=int)
    unknownCount = sp.lil_matrix((1,1),dtype=int) #np.zeros((1,1), dtype = int)
    bondFECounts = [[RNARNACount, DNADNACount, RNADNACount, terminalAUATCount, unknownCount]]
    stemFECounts = bondFECounts*numStems


    for stemNumber, stem in enumerate(STableStructure):
        numBonds = int(len(stem)/2)
        for j in range(numBonds - 1):
            firstNtIndex = stem[j] #the 5' nt (its position in the sequence)
            firstBPIndex = stem[j+numBonds] #what it's bonded to
            secondNtIndex = firstNtIndex + 1 #the 3' nt
            secondBPIndex = firstBPIndex - 1 #what it's bonded to.
                #we're here assuming antiparallel stem. Otherwise, change to + 2*isParallel - 1

            index, bpType = freeEnergyMatrixIndices(sequenceInNumbers,firstNtIndex,firstBPIndex,
                                                        secondNtIndex,secondBPIndex, bound = [1,1],
                                                        unmatchedBPPenalty = unmatchedBPPenalty)

            stemFECounts[stemNumber][bpType][index[0], index[1]] += 1


            stemEnergies[stemNumber] += energyMatrices[bpType][index[0], index[1]]
            stemEntropies[stemNumber] += entropyMatrices[bpType][index[0], index[1]]


        for j in [0, numBonds - 1]: #penalties apply to ends of helices
            if sequenceInNumbers[stem[j]] == 4 or sequenceInNumbers[stem[j + numBonds]] == 4:
                stemFECounts[stemNumber][3][0,0] += 1 #terminal AU/GU was found
            elif sequenceInNumbers[stem[j]] == 8 or sequenceInNumbers[stem[j + numBonds]] == 8:
                stemFECounts[stemNumber][3][0,1] += 1 #terminal AT was found

        if not considerAllAsTerminalMismatches:
            for j in [0, numBonds - 1]: #penalties apply to ends of helices
                if sequenceInNumbers[stem[j]] == 4 or sequenceInNumbers[stem[j + numBonds]] == 4:
                    stemEnergies[stemNumber] += terminal_AU_penalty_energy
                    stemEntropies[stemNumber] += terminal_AU_penalty_entropy
                elif sequenceInNumbers[stem[j]] == 8 or sequenceInNumbers[stem[j + numBonds]] == 8:
                    stemEnergies[stemNumber] += terminal_AT_penalty_energy
                    stemEntropies[stemNumber] += terminal_AT_penalty_entropy

    return(stemEnergies, stemEntropies)


def isComplementary(x,y):
#    x,y define two ntds with the code RNA: A=1,C=2,G=3,U=4; DNA: A=5,C=6,G=7,T=8
#    unknown = 0
    x, y = sorted([x,y])
    if x == 1 and y == 4:
        return(True)
    elif x == 5 and y == 8:
        return(True)
    elif x == 1 and y == 8:
        return(True)
    elif x == 4 and y == 5:
        return(True)
    elif x == 2 and y == 3:
        return(True)
    elif x == 6 and y == 7:
        return(True)
    elif x == 2 and y == 7:
        return(True)
    elif x == 3 and y == 6:
        return(True)
    elif x == 3 and y == 4:
        return(True)
    #Can't have DNA G bind to RNA U

    return(False)


def createSTable(sequence):

    numNt = len(sequence)

    seqInNum  = np.array([1 if sequence[i] == 'A' else
                              2 if sequence[i] == 'C' else
                              3 if sequence[i] == 'G' else
                              4 if sequence[i] == 'U' else
                              0 for i in range(numNt)])

    minBPInStem = 2
    minNtsInHairpin = 3
    substems = 'all'
    maxNumStems = 10**4 #to preallocate space for STable.
    STableStructure = [None]*maxNumStems #first column of STable
    STableBPs = [None]*maxNumStems #second column of STable
    onlyConsiderSubstemsFromEdges = True
    onlyAllowSubsetsOfLongestStems = True

    B = np.zeros((numNt,numNt)) #matrix describing which bases are allowed to bond to which others
    for i in range(numNt):
        for j in range(numNt):
                if isComplementary(seqInNum[i], seqInNum[j]):
                    B[i,j] = 1
                    B[j,i] = 1

    numStems = 0


    BPsAlreadyConsidered = [] #so we avoid subsets of stems until we consider them explicitly later
    maxI = numNt - 2*minBPInStem - minNtsInHairpin + 1 #we need at least minBPInStem consecutive base pairs binding

    for i in range(maxI):
        minJ = i + 2*minBPInStem + minNtsInHairpin - 2
        for j in range(numNt - 1, minJ, -1): #range(start,stop,step)
                #so for example, if we have numNt = 7, minBPInStem = 2, minNtsInHairpin = 3, the only possibility
                #is a hairpin with 0 bonded to 6, and 1 to 5
            if B[i,j] == 1: #if they can bond

                if [i,j] not in BPsAlreadyConsidered:
#                        BPsAlreadyConsidered.append([i,j]) #unnecessary since we definitely won't consider this bp again

                    currentStemI = [i]
                    currentStemJ = [j]
                    listOfPairs = [[i,j]]

                        #now try to lengthen the stem, to include nts i + lenStem and j - lenStem
                    lenStem = 0 #lenStem is one less than the number of bps in a stem.
                    endOfStem = False
                    while not endOfStem:
                        lenStem += 1
                        newI = i + lenStem
                        newJ = j - lenStem
                        if (newI > numNt - 1 or newJ < 0 or #if we've gone beyond the edges of the RNA
                            newJ - newI <= minNtsInHairpin or #or the bps are too close together
                            B[newI, newJ] == 0): #or the bps can't actually bind

                            endOfStem = True
                            lenStem -= 1 #to correct for adding one earlier
                        else:
                            currentStemI.append(newI)
                            currentStemJ.append(newJ)
                            listOfPairs.append([newI,newJ])
                            BPsAlreadyConsidered.append([newI, newJ])
                        #now that we've finished making the longest possible stem starting with the base pair i,j
                        #add that stem to STable
                    if len(currentStemI) >= minBPInStem:
                        STableStructure[numStems] = currentStemI + currentStemJ
                        STableBPs[numStems] = listOfPairs
                        numStems += 1

    if onlyAllowSubsetsOfLongestStems:
            #remove all stems but the longest stems so that in the next step we add
            #only the subsets of the longest stems

        maxLengthStem = minBPInStem
        for i in range(numStems):
            if len(STableStructure[i])/2 > maxLengthStem:
                    maxLengthStem = len(STableStructure[i])/2

        minMaxLengthStem = 16 #don't chop off stems of length more than minMaxLengthStem
        maxLengthStem = min(maxLengthStem,minMaxLengthStem);
        STableBPs = [STableBPs[i] for i in range(numStems) if len(STableStructure[i])/2 >= maxLengthStem]
        STableStructure = [STableStructure[i] for i in range(numStems) if len(STableStructure[i])/2 >= maxLengthStem]
        numStems = len(STableStructure)

            #since we preallocate space, need to make more empty room since we just removed all extra space
        STableBPs += [None]*maxNumStems
        STableStructure += [None]*maxNumStems


    for i in range(numStems):
        fullStemI = STableStructure[i] #the full stem we're considering
        lenStem = int(len(fullStemI)/2)

            #What substems should we consider? This is given by the substems argument to the code
        if substems == 'all':
            minBPInStemSub = minBPInStem
        else: #then substems is an integer
            minBPInStemSub = max(lenStem - substems,minBPInStem)

            #we can make substems of length lengthStem-1 till minBPInStem.
        for j in range(1,lenStem-minBPInStemSub+1): #There are j possible lengths.
                #j also tells us how much shorter the substem is compared to the full stem.
            possSubstemCounters = np.arange(j+1)
            if onlyConsiderSubstemsFromEdges:
                possSubstemCounters = [0,j]

            for k in possSubstemCounters: #substems come from getting rid of either edge.
                truncatedStemI = fullStemI[k:lenStem-j+k] + fullStemI[lenStem+k:len(fullStemI)-j+k]
                    #truncatedStemI is the truncated stem. Add it to STable
                currentStemI = truncatedStemI[:int(len(truncatedStemI)/2)]
                currentStemJ = truncatedStemI[int(len(truncatedStemI)/2):]
                STableStructure[numStems] = currentStemI + currentStemJ

                listOfPairs = [[currentStemI[0],currentStemJ[0]]]
                for l in range(1,len(currentStemI)):
                    listOfPairs.append([currentStemI[l],currentStemJ[l]])
                STableBPs[numStems] = listOfPairs
                numStems += 1

        #remove preallocated space
    STableBPs = [STableBPs[i] for i in range(numStems)]
    STableStructure = [STableStructure[i] for i in range(numStems)]

    return(seqInNum, numStems, STableStructure, STableBPs)

def makeCompatibilityMatrix(numStems, numSequences, STableStructure, STableBPs, frozenStems):

    minNtsInHairpin = 3
    frozenStems = frozenStems
    allowPseudoknots = True

    C = np.zeros((numStems,numStems), dtype = bool)
    for i in range(numStems):
        for j in range(i,numStems):
            if i == j:
                C[i,j] = 1
                C[j,i] = 1
            elif not twoListsShareElement(STableStructure[i],STableStructure[j]):
                C[i,j] = 1
                C[j,i] = 1
                disallowPseudoknotsIJ = not allowPseudoknots
                if numSequences > 1:
                    if linkedStems[i] or linkedStems[j]:
                        disallowPseudoknotsIJ = False

                if disallowPseudoknotsIJ:

                    a = STableStructure[i][0]
                    b = STableStructure[i][int(len(STableStructure[i]/2))]
                    c = STableStructure[j][0]
                    d = STableStructure[j][int(len(STableStructure[j]/2))]

                    if c < a: #switch around labels so we have c>a
                        a = STableStructure[j][0]
                        b = STableStructure[j][int(len(STableStructure[j]/2))]
                        c = STableStructure[i][0]
                        d = STableStructure[i][int(len(STableStructure[i]/2))]

                    if (a<c and c<b and b<d):
                        C[i,j] = 0
                        C[j,i] = 0

                    #make sure each hairpin has at least minBPInHairpin unpaired bps in it.
                    #Right now, this only constrains pairwise compatible regions, but it's a start
                iHairpin = list(range(STableStructure[i][int(len(STableStructure[i])/2) - 1] + 1,
                                         STableStructure[i][int(len(STableStructure[i])/2)]))
                    #The unpaired nts between the start and end of stem i
                jHairpin = list(range(STableStructure[j][int(len(STableStructure[j])/2) - 1] + 1,
                                         STableStructure[j][int(len(STableStructure[j])/2)]))
                    #same for stem j

                    #if the number of unpaired nts is less than minNtsInHairpin
                if (len(np.setdiff1d(iHairpin,STableStructure[j])) < minNtsInHairpin or
                    len(np.setdiff1d(jHairpin,STableStructure[i])) < minNtsInHairpin):
                    C[i,j] = 0
                    C[j,i] = 0

        #That does it for the basic compatibility matrix.

    if frozenStems:
        for i in range(numStems): #for each stem
            if C[i,i]: #not important here, but useful when we repeat this process after the next block.
                for j in range(len(frozenStems)): #for each set of stems that need to be included
                    compatibleWithFrozen = False #is stem i compatible with at least one of the stems
                        #out of the j'th list in frozenStems? It needs to be (for all j) to be included in any structure.
                    for k in range(len(frozenStems[j])):
                        if C[i,frozenStems[j][k]]:
                            compatibleWithFrozen = True #it's compatible with at least one of the stems
                            break

                    if not compatibleWithFrozen: #if for any set of regions one of which needs to be included,
                            #region i isn't compatible with any of them, we can't include it in our structure
                        C[i,:] = 0
                        C[:,i] = 0

    for i in range(numStems):
        for j in range(i+1,numStems):
            if C[i,j]: #not worth going through this code if we already know the regions aren't compatible
                combinedBPs = STableBPs[i] + STableBPs[j]
                combinedStem = bpsList2structure(combinedBPs)
                if len(combinedStem) == 1: #then the combined stems form one single stem
                    C[i,j] = 0
                    C[j,i] = 0

    if frozenStems:
        for i in range(numStems): #for each stem
            if C[i,i]:
                for j in range(len(frozenStems)): #for each set of stems that need to be included
                    compatibleWithFrozen = False #is stem i compatible with at least one of the stems
                        #out of the j'th list in frozenStems? It needs to be (for all j) to be included in any structure.
                    for k in range(len(frozenStems[j])):
                        if C[i,frozenStems[j][k]]:
                            compatibleWithFrozen = True #it's compatible with at least one of the stems
                            break

                    if not compatibleWithFrozen: #if for any set of regions one of which needs to be included,
                            #region i isn't compatible with any of them, we can't include it in our structure
                        C[i,:] = 0
                        C[:,i] = 0

    return(C)


# HELPER FUNCTIONS USED IN THE GILLESPIE ALGORITHM

def calculateTotalFlux(rates):
    totalFlux = 0
    for r in rates:
        totalFlux += r[0]
    return totalFlux

def calculateStemRates(values, kB, T, kind):
    k_0 = 1.0 #adjustable constant
    transitionRates = []
    if kind:
        # we are calculating the rates of forming stems, i.e.,
        # exp(-dS/kB T)
        for i in range(len(values)):
            rate = [k_0 * np.exp((-1)*abs(values[i])/ kB), 1, i]
            transitionRates.append(rate)
    else:
        # we are calculating the rate of breaking a stem
        # exp(- dG/ kB T)
        for i in range(len(values)):
            rate = [k_0*np.exp((-1)*abs(values[i])/(kB*T)), 0, i]
            transitionRates.append(rate)
    return transitionRates

def normalize(rates):
    normalization = sum([rates[i][0] for i in range(len(rates))])
    for i in range(len(rates)):
        rates[i][0] = rates[i][0]/normalization
    return(rates)

def partialSum(rates):
    partial= 0
    for i in range(len(rates)):
        partial += rates[i][0]
    return partial

def LegendreTransform(enthalpies, entropies, T):
    # Need to legendre transform the enthalpy to the gibbs free energy
    # dG = dH - TdS
    gibbsFreeEnergies = []

    for i in range(len(enthalpies)):
        gibbs = enthalpies[i] - T * entropies[i]
        gibbsFreeEnergies.append(gibbs)

    return gibbsFreeEnergies

def findStem(index, possibleStems):
    for i in range(len(possibleStems)):
        if index == possibleStems[i][1]:
            return possibleStems[i]
    return('Error: Could not find this stem!')


def findNewStems(stemsInCurrentStructure, allPossibleStems, C, condition):

    # currentStructure: list of stems in the the current structure
    # allPossibleStems: list of al possible stems for this sequence
    # C: matrix containing the compatibility of two stems
    # stemIndex: the index of the stem we are trying to add
    nextPossibleStems = [] # empty array

    if condition:
        for i in range(len(allPossibleStems)):
            for j in range(len(stemsInCurrentStructure)):
                if i not in stemsInCurrentStructure:
                    if C[i, j] and i !=j:
                        nextPossibleStems.append([allPossibleStems[i], i])

    else:
        for i in range(len(allPossibleStems)):
            # pick a stem
            if i not in stemsInCurrentStructure:
                # if this stem is not already in the structure, we could potentially add it
                canAdd = 0
                for j in range(len(stemsInCurrentStructure)):
                    index = stemsInCurrentStructure[j]
                    if C[i, index] and i != index:
                        canAdd = 0
                    else:
                        canAdd +=1
                if canAdd == 0:
                    nextPossibleStems.append([allPossibleStems[i], i])

    return(nextPossibleStems)


def totalEntropyPerStructure(loop, bond, duplex):
    totalEntropy = []
    for i in range(len(loop)):
        val = loop[i] + bond[i] + duplex[i]
        totalEntropy.append(val)
    return totalEntropy

def structure2stem(structures, listOfStems):
    # converts the structure notation into stem notation that we will need:

    #structure will be array of the form [1, 2, 3], where 1, 2, and 3 correspond to the
    #indices in the array of listOfStems
    ListOfStructures = []
    for i in range(len(structures)):
        structure = []
        for j in range(len(structures[i])):
            structure.append(listOfStems[structures[i][j]])
        ListOfStructures.append(structure)
    return ListOfStructures

def flattenStructure(structure):
    struct = []
    for i in range(len(structure)):
        for j in range(len(structure[i])):
            struct.append(structure[i][j][0])
            struct.append(structure[i][j][1])
    return sorted(struct)

def findTrialStructureRate(trialStructure, allStructures, totalEntropies):
    flatTrialStruct = flattenStructure(trialStructure)
    for i in range(len(allStructures)):
        flatten = flattenStructure(allStructures[i])
        #check = [flatten.count(element) for element in flatten]
        #if all(x <= 1 for x in check):
        if flatTrialStruct == flatten:
            return totalEntropies[i]
    return('Error')

def makeTrialStructures(currentStructure, possibleStems):
    trialStructures = []
    trialIndex = []

    for i in range(len(possibleStems)):

        trial = currentStructure.copy()
        trial.append(possibleStems[i][0])
        trialStructures.append(trial)
        trialIndex.append(possibleStems[i][1])
    return trialStructures, trialIndex


def updateReactionRates(trialStructures, trialIndex, allStructures, totalEntropies, kB=0.0019872):
        # this needs to be changed to actually calculate the transition rates to the next stat
        # self.nextPossibleStems[i] = [ stem, index]
        # self.currentStructure = list of stems in the current structure
        # self.
    updateRates = [] # array of rates

    for i in range(len(trialStructures)):
        trial = trialStructures[i]
         # the stem that we want to find# the index of the stem
        rateOfTrialStructure = findTrialStructureRate(trial, allStructures, totalEntropies)
        if rateOfTrialStructure == 'Error':
            print('Error! Could not find the entropy of the trial structure')
        else:
            entropicRate = [np.exp((-1)*abs(rateOfTrialStructure)/kB), 1, trialIndex[i]]
            updateRates.append(entropicRate)

    return updateRates

def findRate(index, rates):
    for i in range(len(rates)):
        if rates[i][2] == index:
            return rates[i]
