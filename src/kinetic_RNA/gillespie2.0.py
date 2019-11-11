import numpy as np
import kineticFunctions as kF

###################### Gillespie Algorithm at the Stem level ###################
# Author: Harrison LaBollita
# Version: 2.0.0
# Date: November 10, 2019
################################################################################


class Gillespie:
    # initialize
    # sequence: the rna sequence we would like to find the secondary structure off
    # frozenBPs: stems that must be included in the final result
    # cutoff: arbitraty time to stop the gillespie algorithm

    def __init__(self, sequence, frozenBPs, cutoff):

        # initialize sequence to create energies, entropies and stem possiblities from
        # kineticFunctions library

        self.sequence = sequence
        self.frozenBPs = frozenBPs
        self.STableBPs, self.compatibilityMatrix, self.stemEnergies, self.stemEntropies = self.initialize(sequence)

        # intialize the current structure arrays
        self.currentStructure = []

        # initial starting values for the flux, time, and cutoff
        self.totalFlux = 0
        self.cutoff = cutoff
        self.time = 0

        self.possibleStems = []
        self.possibleRates = []
        self.possibleBreakRates = []

    def initialize(self, sequence):
        # We call functions from kineticFunctions, which were taken from
        # Kimichi et. al to produce the stem energies and entropies, as well as,
        # enumerate the possible stems for a given sequence and the compatibiliy
        # matrix, which containes the information on whether stem i is compatible
        # with stem j. For more information see Kimich et. al (https://www.biorxiv.org/content/10.1101/338921v1)

        sequenceInNumbers, numStems, STableStructure, STableBPs, = kF.createSTable(sequence)
        compatibilityMatrix = kF.makeCompatibilityMatrix(numStems, 1, STableStructure, STableBPs)
        stemEnergies, stemEntropies = kF.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
        return(STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)


    def MonteCarloStep(self):
        # Following Dykeman 2015 (https://academic.oup.com/nar/article/43/12/5708/2902645)
        # If we are making the first move then the current structure will have zero length

        # if frozenStems:
        # I will write the code to make this move happen first if the user
        # has specified a frozen stem that must be included in the structure

        if len(self.currentStructure) == 0:

            C = self.compatibilityMatrix
            r1 = np.random.random()
            r2 = np.random.random()

            self.rates = kF.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            self.ratesBreak = kF.calculateStemRates(self.stemEnergies, kB = 0.0019872, T = 310.15, kind = 0)

            self.totalFlux = kF.calculateTotalFlux(self.rates)
            self.time = abs(np.log(r2)/self.totalFlux)

            # loop through all of the rates and partially sum them until we
            # reach our Monte Carlo like condition
            # Note: i = index of the i'th stem
            for i in range(len(self.rates)):
                trial = kF.partialSum(self.rates[:i])

                if trial >= r1 * self.totalFlux:
                    nextMove = self.STableBPs[i]  # we have met the conddition so this stem will be our first move
                    self.currentStructure.append(nextMove)

                    for m in range(len(self.STableBPs)):
                        if C[m, i]:
                            if self.STableBPs[m] not in self.possibleStems:
                                self.possibleStems.append(self.STableBPs[m])
                                self.possibleRates.append(self.rates[m])
                                self.possibleBreakRates.append(self.ratesBreak[m])
                    # we now need to append the possibility of breaking the stem, we just created so
                    self.possibleRates.append(self.ratesBreak[i])
                    self.MemoryOfPossibleStems = self.possibleStems
                    # Question here!!!!
                    self.totalFlux = r1*self.totalFlux - kF.partialSum(self.possibleRates[:i]) # recalculate the partial sum
                    print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), str(self.currentStructure)))
                    break

        else:
        # after making the first stem or initial frozen stems we will always be in this condition

            r1 = np.random.random()
            r2 = np.random.random()
            self.time += abs(np.log(r2)/self.totalFlux)

        # Now we will only consider the possible rates that can form with our first chosen stem

            for i in range(len(self.possibleRates)):

                trial = kF.partialSum(self.possibleRates[:i])

                if trial >= r1*self.totalFlux:

                    if i >= len(self.possibleStems):
                        break

                    if self.possibleRates[i][1]:
                    # This means that we have chosen a rate that corresponds to forming this stem
                        nextMove = self.possibleStems[i]
                        if kF.canAdd(self.currentStructure, nextMove):
                            self.currentStructure.append(nextMove)
                            self.possibleRates.append(self.possibleBreakRates[i]) # now append the possiblity of breaking this stem

                            del self.possibleRates[i] # remove this rate from happening
                            del self.possibleStems[i] # remove this stem because now it has been chosen

                            self.totalFlux = r1*self.totalFlux - kF.partialSum(self.possibleRates[:i])
                            print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), str(self.currentStructure)))
                            break

                    else:
                    # We have chosen to break the stem so we will remove it from the current structure
                    # self.possibleRates[i][0] will match one of the rates in self.possibleRatesBreak

                        for j in range(len(self.possibleRatesBreak)):
                            if self.possibleRates[j][0] == self.possibleRatesBreak:
                            # found the rate that corresponds with this stem
                                breakThisStem = self.MemoryOfPossibleStems[i]
                                for k in range(len(self.currentStructure)):
                                    if self.currentStructure[k] == breakThisStem:
                                        del self.currentStructure[k]
                                    else:
                                        print('Error: Move chosen to break stem %s, but can not find this stem in the current structure' %(str(breakThisStem)))
                        del self.possibleRates[i]
                        print('Time: %0.2fs | Broke Stem: %s | Current Structure: %s' %(self.time, str(breakThisStem), str(self.currentStructure)))
                        self.totalFlux = r1*self.totalFlux - kF.partialSum(self.possibleRates[:i])
                        break

        return(self)

    def runGillespie(self):
        while self.time < self.cutoff:
            self.MonteCarloStep()
        return(self.currentStructure)

G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 1000)
structure = G.runGillespie()
