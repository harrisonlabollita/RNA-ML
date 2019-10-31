from RFE_landscape import RNALandscape
import numpy as np
import copy
import scipy

########################## GILLESPIE ALGORITHMN ################################
# Initialization
#     - Given sequence of RNA
#     - Use RFE_landscpae to initialize:
#                                     all possible stems,
#                                     transition rates
#                                     and probabilities
#     - Need to verify a couple of things:
#
#  2. Monte Carlo Step
#    - Use random numbers to determine the next possible move to make based off of probaility and transition rate
#    - Move must be compatible with  the current structure (S_0)
# 3. Update
#    - Save the move
#    - Update structure S_0 ---> S_1 = stem1 + stem2
#    - Time += time_step
# 4. Iterate
#    - Repeat Monte Carlo step until simulation time runts out
################################################################################

############################## HELPER FUNCTIONS ################################
# Functions taken from RFE Landscape and used independuntly from the calculate
# FE landscapre funciton in the RFE class.

def bondFreeEnergiesRNARNA():
    #define the RNA/RNA bond enthalpy and entropy arrays

    #Sources: Table 4 of Xia et al Biochemistry '98
        #Table 4 of Mathews et al. JMB '99
        #Table 3 of Xia, Mathews, Turner "Thermodynamics of RNA secondary structure formation" in book by Soll, Nishmura, Moore
    #First index tells you if the first bp of the set is AU (0) CG (1) GC (2) UA (3) GU (4) or UG (5)
    #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or U(3) (which row of table 1 in Serra & Turner).
    #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or U(3) (which column of table 1 in Serra & Turner).

#    Had to make this 2D array to be able to use scipy sparse functions, so second/third index is replaced
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
    #define the DNA/DNA bond enthalpy and entropy arrays

    #Data is from From: Thermodynamics and NMR of Internal G·T Mismatches in DNA and other papers by Allawi and
    #various SantaLucia publications (cited as 28-32 in Mfold web server for nucleic acid folding and hybridization prediction)
    #First index tells you if the first bp of the set is AT (0) CG (1) GC (2) or TA (3)
    #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or T(3)
    #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or T(3)

#    Had to make this 2D array to be able to use scipy sparse functions, so second/third index is replaced
#    by (4*second index) + third index

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
    #define the RNA/DNA bond enthalpy and entropy arrays

    #First index tells you if the set is (putting RNA first)
    #5'AX3'/3'TY5' (0) 5CX3/3GY5 (1) 5GX3/3CY5 (2) 5UX3/3AY5 (3) 5XA3/3YT5 (4) 5XC3/3YG5 (5) 5XG3/3YC5 (6) or 5XU3/3YA5 (7).
    #Second index tells you if X is A (0) C (1) G (2) or U (3)
    #Third index tells you if Y is A (0) C (1) G (2) or T (3)
    #Data is from From: Thermodynamic Parameters To Predict Stability of RNA/DNA Hybrid Duplexes &
    #Thermodynamic contributions of single internal rA·dA, rC·dC, rG·dG and rU·dT mismatches in RNA/DNA duplexes

#    Had to make this 2D array to be able to use scipy sparse functions, so second/third index is replaced
#    by  (4*second index) + third index

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
# =============================================================================
#     Get the indices of the energy/entropy matrix to use for
#    the base pair stack:
#
#    5' firstNt, secondNt 3'
#    3' firstBP, secondBP 5'
#
#    where firstNt is sequenceInNumbers[firstNtIndex], etc.
#
#    Assumes firstNt and firstBP are actually bound, but secondNt and secondBP need not be
#    bound = [are_firstNt_and_firstBP_bound, are_secondNt_and_secondBP_bound]
#    unmatchedBPPenalty = True if two nts that could be bound but aren't should be treated as
#    an A-C pair. On top of whether or not we treat them as an A-C pair, we introduce an
#    [energy, entropy] penalty given by unboundButCouldBindPenalties
# =============================================================================
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
        #For the bondEnergyMatrix. We're dealing with DNA/DNA pair
        #First index tells you if the first bp of the set is AT (0) CG (1) GC (2) or TA (3)
        #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or T(3)
        #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or T(3)
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
        #For the bondEnergyMatrix,
#        First index tells you if the set is (putting RNA first) (0) 5'AX3'/3'TY5' (1) 5CX3/3GY5
#       (2) 5GX3/3CY5 (3) 5UX3/3AY5 (4) 5XA3/3YT5 (5) 5XC3/3YG5 (6) 5XG3/3YC5 or (7) 5XU3/3YA5.

        #firstNt and secondNt are fixed to be RNA while firstBP and secondBP are DNA. Therefore the order of which
        #base pair is first or second is in some cases switched (which is why we may have had to flip the stem
#        upside down a minute ago)
        #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or U(3)
        #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or U(3)

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
        #need -1 for RNA nts since nts are numbered 1-4 but indices are 0-3
        #need -5 for DNA nts since nts are numbered 5-8 but indices are 0-3
#        Had to make the energy/entropy matrices 2D which is why secondNt and secondBP are
#        not being returned as separate indices, but as one index (4*secondNt + secondBP)
    def checkCompatibility(C, C3, C4, numStems, linkedStems):
# =============================================================================
#        Make the list of structures. Each structure is defined by the list of stems that comprise it
#        (previously helipoints)
# =============================================================================

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
# =============================================================================
#         Calculate the bond energy and entropy of each stem, not including what to do at the
#        ends of the stems (i.e. dangling ends, terminal mismatches, and the like)
# =============================================================================
    unmatchedBPPenalty = True
    includeTerminalAUATPenalties = True
    considerAllAsTerminalMismatches = False
#        #Define energy (enthalpy; deltaH; units of kcal/mol) and entropy (deltaS; units of kcal/(mol*K)) matrices for bonds.
    energyMatrices, entropyMatrices = bondFreeEnergies()
#
    stemEnergies = np.zeros(numStems)
    stemEntropies = np.zeros(numStems)

    if includeTerminalAUATPenalties:
            (terminal_AU_penalty_energy, terminal_AU_penalty_entropy,
             terminal_AT_penalty_energy, terminal_AT_penalty_entropy) = terminalATAUPenalties()
    else:
            (terminal_AU_penalty_energy, terminal_AU_penalty_entropy,
             terminal_AT_penalty_energy, terminal_AT_penalty_entropy) = [0,0,0,0]


    RNARNACount = scipy.sparse.lil_matrix((6,16),dtype=int) #np.zeros((6,4,4),dtype = int)
        #First index tells you if the first bp of the set is AU (0) CG (1) GC (2) UA (3) GU (4) or UG (5)
        #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or U(3) (which row of table 1 in Serra & Turner).
        #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or U(3) (which column of table 1 in Serra & Turner).

#    Had to make this and the others 2D array to be able to use scipy sparse functions, so
#    second/third index is replaced by (4*second index) + third index. That's why it's
#        np.zeros(6,16) and not np.zeros(6,4,4).
#        lil_matrix was chosen because, from scipy reference page, it is fast for constructing
#        sparse matrices incrementally. For operations (later we'll multiply it) it should be converted to
#        another form.

    DNADNACount = scipy.sparse.lil_matrix((4,16),dtype=int) #np.zeros((4,16),dtype = int)
        #First index tells you if the first bp of the set is AT (0) CG (1) GC (2) or TA (3)
        #Second index tells you if the 3' ntd of the second bp is A (0) C (1) G(2) or T(3)
        #Third index tells you if the 5' ntd of the second bp is A (0) C (1) G(2) or T(3)

    RNADNACount = scipy.sparse.lil_matrix((8,16),dtype=int) #np.zeros((8,16),dtype = int)
        #First index tells you if the set is (putting RNA first)
        #5'AX3'/3'TY5' (0) 5CX3/3GY5 (1) 5GX3/3CY5 (2) 5UX3/3AY5 (3) 5XA3/3YT5 (4) 5XC3/3YG5 (5) 5XG3/3YC5 (6) or 5XU3/3YA5 (7).
        #Second index tells you if X is A (0) C (1) G (2) or U (3)
        #Third index tells you if Y is A (0) C (1) G (2) or T (3)

    terminalAUATCount = scipy.sparse.lil_matrix((1,2),dtype=int)
#        Number of terminal AU (or GU) pairs, number of terminal AT pairs

    unknownCount = scipy.sparse.lil_matrix((1,1),dtype=int) #np.zeros((1,1), dtype = int)

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


# =============================================================================
#    Include terminal penalties for AU, GU, or AT pairs at the ends of helices
#    (from Xia et al. 1998 for RNA, or SantaLucia and Hicks (2004) for DNA):
#    GU penalty is assumed to be same as for AU (Xia paper, or NNDB)
#    From Xia, Mathews, Turner paper (from Soll, Nishimura, Moore book):
#    "Note that when dangling ends or terminal mismatches follow terminal AU
#    or GU pairs, the penalty for a terminal AU is still applied"
#
#    Guess that  this penalty also applies for DNA/RNA interactions since
#    it's physically motivated by the number of H-bonds AU/AT pairs have
#     if (bpType == 0 and index[0] == 0) or (bpType == 2 and (index[0] == 3 or index[0] == 7)):
#         terminalAUPenaltyCounter += 1
#     elif (bpType == 1 and index[0] == 0) or (bpType == 2 and (index[0] == 0 or index[0] == 4)):
#         terminalATPenaltyCounter += 1
# =============================================================================
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
