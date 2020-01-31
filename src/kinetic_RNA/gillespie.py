######## GILLESPIE ALGORITHM AT STEM LEVEL FOR RNA STRUCTURE PREDICTION ########

#################################################################################
# Author: Harrison LaBollita                                                    #
# Advisor: Petr Sulc                                                            #
# Date: December 6, 2019                                                        #
# Most of calculation is built upon the work of Kimich et. al                   #
# (https://www.biorxiv.org/content/10.1101/338921v1). The implementation of the #
# Gillespie algorithm was done following Dykeman                                #
# (https://academic.oup.com/nar/article/43/12/5708/2902645).                    #
#################################################################################

# This code depends on a helper functions file title 'helperfunctions.py' to do
# some of the mundane routines throughout the algorithm, i.e., searches and sums.
# In addition, when the program initializes we use the free energy landscape
# algorithm produced by Kimichi. Note that this program is very slow, and is not
# optimized for its ideal use which are longer RNA sequences. However, our
# algorithm has a significant advantage to all other RNA kinetic folders and that
# is the ability to handle pseudoknots.


import numpy as np
import helperfunctions as hf
import RFE_landscape as RFE
import time
class Gillespie:

    def __init__(self, sequence, constraints, maxTime, toPrint = True, initTime = True):

        # We begin by initializing all the components that we will need throughout the algorithm.
        # This includes all possible stems, structures, entropies, and gibbs free energy.

        self.sequence = sequence # sequence that we will fold
        self.constraints = constraints # if we have a frozen stem that we would like to include in the final calculation
        if initTime:
           start = time.time()
           self.allPossibleStems, self.STableStructure, self.compatibilityMatrix, self.allStructures, self.allStructures2, self.stemEnergies, self.stemEntropies, self.totalEntropies = self.initialize(sequence)
           stop = time.time()
           self.initializationTime = stop - start
        else:
           self.allPossibleStems, self.STableStructure, self.compatibilityMatrix, self.allStructures, self.allStructures2, self.stemEnergies, self.stemEntropies, self.totalEntropies = self.initialize(sequence)
        self.allPossibleStems2 = [ [self.allPossibleStems[i], i] for i in range(len(self.allPossibleStems))]

        # need to convert the enthalpy to the gibbs free energy
        self.stemGFEnergies = hf.LegendreTransform(self.stemEnergies, self.stemEntropies, 310.15)

        # intialize the current structure arrays
        self.currentStructure = []
        self.stemsInCurrentStructure = []

        # initial starting values for the flux, time, and cutoff
        self.totalFlux = 0
        self.maxTime = maxTime
        self.time = 0

        self.nextPossibleStems = [] #initialize
        self.nextPossibleRates = [] #initialize

        self.toPrint = toPrint
        self.kB = 0.0019872

    def initialize(self, sequence):
        # See Kimichi et. al (https://www.biorxiv.org/content/10.1101/338921v1)

        # Call RNALandscape to initialize all the quantities that we will need throughtout our algorithm

        q = RFE.RNALandscape([sequence])
        q.calculateFELandscape()

        sequenceInNumbers = q.sequenceInNumbers
        numStems = q.numStems
        STableStructure = q.STableStructure
        STableBPs = q.STableBPs
        compatibilityMatrix = q.C
        stemEnergies, stemEntropies = hf.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
        allLoopEntropies = q.allLoopEntropies
        allBondEntropies = q.allBondEntropies
        allDuplexEntropies = q.allDuplexEntropies
        allStructures = hf.structure2stem(q.structures, STableBPs)
        allStructures2 = q.structures

        totalEntropies = hf.totalEntropyPerStructure(allLoopEntropies, allBondEntropies, allDuplexEntropies)

        return(STableBPs, STableStructure, compatibilityMatrix, allStructures, allStructures2, stemEnergies, stemEntropies, totalEntropies)
    
    def convert2dot(self, currentStructure):
        # Function to convert the notation of the current structure to dot bracket notation
        # Not written to handle pseudoknots yet
        representation = ''
        dotbracket = [0]*len(self.sequence)
        # find the pseudoknots first and add those in the dotbracket notation
        if len(currentStructure) == 1:
            for i in range(len(currentStructure)):
                for j in range(len(currentStructure[i])):
                    dotbracket[currentStructure[i][j][0]] = 1
                    dotbracket[currentStructure[i][j][1]] = 2
        for i in range(len(currentStructure)):
            firstStem = currentStructure[i]
            for j in range(len(currentStructure)):
                nextStem = currentStructure[j]
                if firstStem != nextStem:
                    for k in range(len(firstStem)):
                        for l in range(len(nextStem)):
                            base1 = firstStem[k][0]
                            base2 = firstStem[k][1]
                            base3 = nextStem[l][0]
                            base4 = nextStem[l][1]

                            if base1 < base3 < base2 < base4:
                                # then we have a pseudoknot
                                dotbracket[base1] = 1
                                dotbracket[base2] = 2
                                dotbracket[base3] = 3
                                dotbracket[base4] = 4
                            else:
                                dotbracket[base1] = 1
                                dotbracket[base2] = 2
                                dotbracket[base3] = 1
                                dotbracket[base4] = 2
        # convert 0's, 1's, 2's 3',and 4's into '.', '(', ')', '[' ']'

        for element in dotbracket:
            if element == 0:
                representation += '.'
            elif element == 1:
                representation += '('
            elif element == 2:
                representation += ')'
            elif element == 3:
                representation += '['
            else:
                representation += ']'
        return(representation)

    def constraintHandler(self):
        # function to handle the constraints given by Menghan
        # discussed the constraints to be a vector of the form [[12, '('], [16, ')'], ... ]
        # Requirements:
        # - only allow for moves that satisfy the constraints
        # - if the move does not satisfy the constraint the script will need to break
        #   and start over creating a new move
        return True



    def MonteCarloStep(self):

        # Begin with any frozen contraints that will need to be considered, but we will add this
        # component last

        # This is our first move!
        if not self.time:
            C = self.compatibilityMatrix #for readability we rename this

            # generate two random numbers
            r1 = np.random.random()
            r2 = np.random.random()

            # calculate the transition rates for all the states, this done using the kineticFunctions file.
            self.allRates = hf.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            self.ratesBreak = hf.calculateStemRates(self.stemGFEnergies, kB = 0.0019872, T = 310.15, kind = 0)
            self.totalFlux = sum([r[0] for r in self.allRates]) # the sum off all the rates
            self.time += abs(np.log(r2))
            #self.time += (abs(np.log(r2))/self.totalFlux) # increment the reaction time for next state
            normalizedRates = hf.normalize(self.allRates) # normalize the rates such that they sum to one


            for i in range(len(normalizedRates)):
                trial = hf.partialSum(normalizedRates[:i])

                if trial >= r1:
                    stateEntropy = self.stemEntropies[i]
                    nextMove = self.allPossibleStems[i]
                    self.currentStructure.append(nextMove)  # append the stem to the current structure
                    self.stemsInCurrentStructure.append(i)  # append the index of this stem into a list to keep track of what stems are coming in and out of current structure
                    # update the user on what move was made
                    if self.toPrint:
                        print('Time: %0.4fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), self.convert2dot(self.currentStructure)))


                    # we now need to calculate the next set of possible moves and
                    # the rates corresponding to these moves

                    for m in range(len(self.allPossibleStems)):
                        if C[i, m] and m != i:
                            self.nextPossibleStems.append([self.allPossibleStems[m], m]) # format of this array will be [stem_m , and m = index of stem from larger array]

                    trialStructures, trialIndices = hf.makeTrialStructures(self.currentStructure, self.nextPossibleStems, self.allStructures, len(self.sequence))
                    self.nextPossibleRates = hf.updateReactionRates(trialStructures, trialIndices, self.allStructures, self.totalEntropies, stateEntropy, len(self.sequence))
                    self.nextPossibleRates.insert(0, self.ratesBreak[i])

                    self.totalFlux = sum([r[0] for r in self.nextPossibleRates])

                    self.nextPossibleRates = hf.normalize(self.nextPossibleRates)
                    return(self)

        else:
        # Now we are in our 2+ move.

        # generate two random numbers
            r1 = np.random.random()
            r2 = np.random.random()

        # update time
            #self.time += (abs(np.log(r2))/self.totalFlux)
            self.time += abs(np.log(r2))
            # find the next move
            for i in range(len(self.nextPossibleRates)):
                trial = hf.partialSum(self.nextPossibleRates[:i])

                if trial >= r1:

                    if self.nextPossibleRates[i][1]: # this will be true if we have chosen to add a stem
                        stateEntropy = self.kB * np.log(self.nextPossibleRates[i][0])
                        index = self.nextPossibleRates[i][2] # the index of the stem that we will add
                        nextMove = hf.findStem(index, self.nextPossibleStems)
                        stemIndex = nextMove[1]

                        self.currentStructure.append(nextMove[0])
                        self.stemsInCurrentStructure.append(stemIndex)
                        if self.toPrint:
                            print('Time: %0.4fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove[0]), self.convert2dot(self.currentStructure)))
                        # check for new stems that could be compatible with the structure
                        self.nextPossibleStems = hf.findNewStems(self.stemsInCurrentStructure, self.allPossibleStems2, self.allStructures2)
                        trialStructures, trialIndices = hf.makeTrialStructures(self.currentStructure, self.nextPossibleStems, self.allStructures, len(self.sequence))
                        self.nextPossibleRates = hf.updateReactionRates(trialStructures, trialIndices, self.allStructures, self.totalEntropies, stateEntropy, len(self.sequence))

                        for ind in self.stemsInCurrentStructure:
                            self.nextPossibleRates.insert(0, hf.findRate(ind, self.ratesBreak))
                        self.totalFlux = sum([r[0] for r in self.nextPossibleRates])
                        self.nextPossibleRates = hf.normalize(self.nextPossibleRates)
                        return(self)

                    else:
                        # we have chosen to break a stem
                        # We will now find the stem to break in our current structure, then populate a list of new
                        # new stems to consider for the next move.
                        stemIndexToRemove = self.nextPossibleRates[i][2]
                        stemToBreak = hf.findStem(stemIndexToRemove, self.allPossibleStems2)
                        stateEntropy = self.kB * np.log(self.nextPossibleRates[i][0])
                        for k in range(len(self.stemsInCurrentStructure)): #searching for the stem to break
                            if stemIndexToRemove == self.stemsInCurrentStructure[k]:
                                del self.currentStructure[k]
                                del self.stemsInCurrentStructure[k]
                                if self.toPrint:
                                    print('Time: %0.4fs | Broke Stem: %s | Current Structure: %s' %(self.time, str(stemToBreak[0]), self.convert2dot(self.currentStructure)))

                                if len(self.currentStructure) == 0:

                                    self.nextPossibleRates = hf.normalize(self.allRates)
                                    self.nextPossibleStems = self.allPossibleStems2
                                    self.totalFlux = sum([r[0] for r in self.nextPossibleRates])
                                else:
                                    self.nextPossibleStems = hf.findNewStems(self.stemsInCurrentStructure, self.allPossibleStems2, self.allStructures2)
                                    trialStructures, trialIndices = hf.makeTrialStructures(self.currentStructure, self.nextPossibleStems, self.allStructures, len(self.sequence))
                                    self.nextPossibleRates = hf.updateReactionRates(trialStructures, trialIndices, self.allStructures, self.totalEntropies, stateEntropy, len(self.sequence))
                                    for ind in self.stemsInCurrentStructure:
                                        self.nextPossibleRates.insert(0, hf.findRate(ind, self.ratesBreak))
                                    self.totalFlux = sum([r[0] for r in self.nextPossibleRates])
                                    self.nextPossibleRates = hf.normalize(self.nextPossibleRates)
                                return(self)

    def runGillespie(self):
        # run the gillespie algorithm until we reach maxTime
        self.MonteCarloStep()
        while self.time < self.maxTime:
            self.MonteCarloStep()
        return(self.convert2dot(self.currentStructure))

    def averageGillespie(self):
        maxIter = 10
        iter = 0
        predictions = []
        while iter < maxIter:
            structure = self.runGillespie()
            predictions.append(structure)

        averages = [[pred, predictions.count(pred)] for pred in predictions]
        max = 0
        for i in range(len(averages)):

            if averages[i][1] >= max:
                answer = averages[i][0]
                max = averages[i][1]
        return answer

#'CGGUCGGAACUCGAUCGGUUGAACUCUAUC'
#UGCCUGGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAACGCCGUAGCGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGGAACUGCCAGGCAU

#G = Gillespie('GGGGACCCCGCGCACCCGCCAGAGCCCGUUGACCCUUGCUGCCUUCCGGCCCUGGGGGAGUUCACAGGAUGGACGCCGCGCGGGGUCC', [], maxTime = 5,toPrint = True)
#structure = G.runGillespie()

################################# EXAMPLE ########################################
#G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], maxTime = 5, toPrint = True)
#structure = G.runGillespie()                                                     #
#print(structure)                                                                #
##################################################################################
