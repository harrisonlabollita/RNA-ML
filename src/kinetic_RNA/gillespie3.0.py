import numpy as np
import kineticFunctions as kF

###################### Gillespie Algorithm at the Stem level ###################
# Author: Harrison LaBollita
# Version: 3.0.0
# Date: November 15, 2019
################################################################################



class Gillespie:
    # initialize
    # sequence: the rna sequence we would like to find the secondary structure off
    # frozenBPs: stems that must be included in the final result
    # cutoff: arbitraty time to stop the gillespie algorithm

    def __init__(self, sequence, frozen, maxTime):

        # initialize sequence to create energies, entropies and stem possiblities from
        # kineticFunctions library

        self.sequence = sequence
        self.frozen = frozen
        self.allPossibleStems, self.compatibilityMatrix, self.stemEnergies, self.stemEntropies = self.initialize(sequence)
        # need to convert the enthalpy to the gibbs free energy

        self.stemGFEnergies = kF.LegendreTransform(self.stemEnergies, self.stemEntropies, 310.15)
        # intialize the current structure arrays
        self.currentStructure = []
        self.stemsInCurrentStructure = []

        # initial starting values for the flux, time, and cutoff
        self.totalFlux = 0
        self.maxTime = maxTime
        self.time = 0

        self.nextPossibleStems = []
        self.nextPossibleRates = []

    def initialize(self, sequence):
        # We call functions from kineticFunctions, which were taken from
        # Kimichi et. al to produce the stem energies and entropies, as well as,
        # enumerate the possible stems for a given sequence and the compatibiliy
        # matrix, which containes the information on whether stem i is compatible
        # with stem j. For more information see Kimich et. al (https://www.biorxiv.org/content/10.1101/338921v1)
        frozenStems = []
        sequenceInNumbers, numStems, STableStructure, STableBPs, = kF.createSTable(sequence)
        compatibilityMatrix = kF.makeCompatibilityMatrix(numStems, 1, STableStructure, STableBPs, frozenStems)
        stemEnergies, stemEntropies = kF.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
        return(STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)

    def MonteCarloStep(self):
        # Following Dykeman 2015 (https://academic.oup.com/nar/article/43/12/5708/2902645)
        # If we are making the first move then the current structure will have zero length

        # Will add in the future, but at the very beginning of the code, we will check to see
        # if there are any stems that must be included in the final result. If the user has
        # specified stems at this point we will treat this as the first reaction and then step forward in time.


        #if frozen:
            # then we have frozen stems are we need to make sure that they are included first
            #r2 = np.random.random()
            #self.rates = kF.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            #self.time = abs(np.log(r2)/totalFlux)
            #self.currentStructure.append(frozen)

            # Need to add the frozen feature piece
            # need to go back and check a few things

        if len(self.currentStructure) == 0:

            C = self.compatibilityMatrix

            # generate two random numbers
            r1 = np.random.random()
            r2 = np.random.random()

            # calculate the transition rates for all the states, this done using the kineticFunctions file.
            self.rates = kF.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            self.ratesBreak = kF.calculateStemRates(self.stemGFEnergies, kB = 0.0019872, T = 310.15, kind = 0)
            self.totalFlux = kF.calculateTotalFlux(self.rates) # the sum off all the rates
            self.time += abs(np.log(r2)) # increment the reaction time for next state
            normalized_rates = kF.normalize(self.rates) # normalize the rates such that the sum to one

            for i in range(len(normalized_rates)):
                trial = kF.partialSum(normalized_rates[:i])
                if trial >= r1:
                    nextMove = self.allPossibleStems[i] # we have met the condition so this stem will be our first move
                    self.currentStructure.append(nextMove)
                    self.stemsInCurrentStructure.append(i)
                    # remove this move from itself
                    for m in range(len(self.allPossibleStems)):
                        if C[i, m] and m != i :
                            # at this point we find all the next possible states for our structure to be in, in other words,
                            # the list of possible stems with our current structure
                            self.nextPossibleStems.append([self.allPossibleStems[m], m]) # we will keep track of the stem and the label for that stem i.e.,
                                                                                         # self.nextPossibleStems[i] = [ stem_i, index] index = m
                    # at this point we would recalculate the stem rates and append the breaking rate
                    if len(self.nextPossibleStems):
                        self.nextPossibleRates = kF.updateReactionRates(self.nextPossibleStems)
                        self.nextPossibleRates.insert(0, self.ratesBreak[i])
                        self.nextPossibleRates = kF.normalize(self.nextPossibleRates)
                    print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), self.convert2dot(self.currentStructure)))
                    break

        else:
        # Now we are in our 2+ move.

        # generate two random numbers
            r1 = np.random.random()
            r2 = np.random.random()
        # update time

            self.time += abs(np.log(r2))
        # find the next move
            for i in range(len(self.nextPossibleRates)):
                trial = kF.partialSum(self.nextPossibleRates[:i])
                if trial >= r1:

                    if self.nextPossibleRates[1]: # this will be true if we have chosen to add a stem
                        index = self.nextPossibleRates[i][2] # the index of the stem that we will add
                        nextMove = kF.findStem(index, self.nextPossibleStems)
                        stemIndex = nextMove[1]
                        self.currentStructure.append(nextMove[0])
                        self.stemsInCurrentStructure.append(stemIndex)
                        # check for new stems that could be compatible with the structure
                        self.nextPossibleStems = kF.findNewStems(self.stemsInCurrentStructure, self.allPossibleStems, self.compatibilityMatrix, stemIndex)
                        # calculate the new rates for the next state
                        self.nextPossibleRates = kF.normalize(kF.updateReactionRates(self.nextPossibleStems))
                        #print(kF.calculateTotalFlux(self.possible Rates))
                        print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove[0]), self.convert2dot(self.currentStructure)))

                    else:
        # We have chosen to break the stem so we will remove it from the current structure
        # self.possibleRates[i][0] will match one of the rates in self.possibleRatesBreak
                        stemIndexToRemove = self.nextPossibleRates[i][2]
                        StemToBreak = kF.findStem(stemIndexToRemove, self.allPossibleStems)
                        for k in range(len(stemsInCurrentStructure)):
                            stemIndexToRemove = stemsInCurrentStructure[i]
                            if stemIndex == stemToRemove:
                                del self.currentStructure[i]
                                self.nextPossibleStems = kF.findNewStems(self.stemsInCurrentStructure, self.allPossibleStems, self.compatibilityMatrix, -1)
                                self.nextPossibleRates = kF.normalize(kF.updateReactionRates(self.nextPossibleStems))
                                print('Time: %0.2fs | Broke Stem: %s | Current Structure: %s' %(self.time, str(stemToBreak), self.convert2dot(self.currentStructure)))
        return(self)

    def convert2dot(self, currentStructure):
        # convert to dot bracket notation including pseudoknots
        representation = ''
        dotbracket = [0]*len(self.sequence)


        for i in range(len(currentStructure)):
            for j in range(len(currentStructure[i])):
                open = currentStructure[i][j][0]
                close = currentStructure[i][j][1]
                dotbracket[open] = 1
                dotbracket[close] = 2
        for element in dotbracket:
            if element == 0:
                representation += '.'
            elif element == 1:
                representation += '('
            else:
                representation += ')'
        return(representation)


    def runGillespie(self):
        while self.time < self.maxTime:
            self.MonteCarloStep()
        return(self.currentStructure)

G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 10)
structure = G.runGillespie()
