import numpy as np
import kineticFunctions as kF

###################### Gillespie Algorithm at the Stem level ###################
# Author: Harrison LaBollita
# Version: 2.0.0
# Date: November 11, 2019
################################################################################

# THIS DOES NOT WORK ANYMORE
class Gillespie:
    # initialize
    # sequence: the rna sequence we would like to find the secondary structure off
    # frozenBPs: stems that must be included in the final result
    # cutoff: arbitraty time to stop the gillespie algorithm

    def __init__(self, sequence, frozen, cutoff):

        # initialize sequence to create energies, entropies and stem possiblities from
        # kineticFunctions library

        self.sequence = sequence
        self.frozen = frozen
        self.STableBPs, self.compatibilityMatrix, self.stemEnergies, self.stemEntropies = self.initialize(sequence)
        # need to convert the enthalpy to the gibbs free energy

        self.stemGFEnergies = kF.LegendreTransform(self.stemEnergies, self.stemEntropies, 310.15)
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
        frozenStems = []
        sequenceInNumbers, numStems, STableStructure, STableBPs, = kF.createSTable(sequence)
        compatibilityMatrix = kF.makeCompatibilityMatrix(numStems, 1, STableStructure, STableBPs, frozenStems)
        stemEnergies, stemEntropies = kF.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
        return(STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)






    def MonteCarloStep(self):
        # Following Dykeman 2015 (https://academic.oup.com/nar/article/43/12/5708/2902645)
        # If we are making the first move then the current structure will have zero length

        # if frozen:
        # I will write the code to make this move happen first if the user
        # has specified a frozen stem that must be included in the structure
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
            #for i in range(len(C)):
            #    for j in range(len(C[i])):
            #        if i == 3:
            #            print('%s at %d, %d' %(C[i,j], i, j))

            r1 = np.random.random()
            r2 = np.random.random()
            self.rates = kF.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            self.ratesBreak = kF.calculateStemRates(self.stemGFEnergies, kB = 0.0019872, T = 310.15, kind = 0)
            #self.totalFlux = kF.calculateTotalFlux(self.rates) # normalize the rates
            self.totalFlux = kF.calculateTotalFlux(self.rates)
            self.time = abs(np.log(r2)/self.totalFlux)

            # Made sure that the rates in fact sum to 1!
            # loop through all of the rates and partially sum them until we
            # reach our Monte Carlo like condition
            # Note: i = index of the i'th stem
            normalized_rates = kF.normalize(self.rates)

            for i in range(len(normalized_rates)):
                trial = kF.partialSum(normalized_rates[:i])

                if trial >= r1:
                    nextMove = self.STableBPs[i] # we have met the condition so this stem will be our first move
                    self.currentStructure.append(nextMove)
                    # remove this move from itself
                    for m in range(len(self.STableBPs)):
                        if C[i, m] and m != i :
                            self.possibleStems.append([self.STableBPs[m], m])
                            rate = self.rates[m]
                            rate.append(m)
                            self.possibleRates.append(rate)
                            rateB = self.ratesBreak[m]
                            rateB.append(m)
                            self.possibleBreakRates.append(rateB)
                            # the rate arrays both have the format, where
                                # self.possibleRates[i][0] = rate
                                # self.possibleRates[i][1] = break or form
                                # self.possibleRates[i][2] = stem that this rate corresponds too
                    if not len(self.possibleStems):
                        print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), self.convert2dot()))
                        break
                    # we now need to append the possibility of breaking the stem, we just created so

                    toAdd = self.ratesBreak[i]
                    toAdd.append(i)
                    self.possibleRates.insert(0, toAdd)
                    print(self.possibleRates)
                    print(self.possibleStems)
                    self.possibleRates = kF.normalize(self.possibleRates) # renormalize these rates appropraitely
                    #print(kF.calculateTotalFlux(self.possibleRates))
                    self.MemoryOfPossibleStems = self.possibleStems
                    # at this point we need to renormalize the rates
                    print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), self.convert2dot()))
                    break

        else:
        # after making the first stem or initial frozen stems we will always be in this condition

            r1 = np.random.random()
            r2 = np.random.random()

            self.time += abs(np.log(r2)/self.totalFlux)

        # Now we will only consider the possible rates that can form with our first chosen stem

            for i in range(len(self.possibleRates)):

                trial = kF.partialSum(self.possibleRates[:i])
                if trial >= r1:
                    if self.possibleRates[i][1]:
                    # This means that we have chosen a rate that corresponds to forming this stem
                        index = self.possibleRates[i][2]
                        nextMove = kF.findStem(index, self.possibleStems)
                        if kF.canAdd(self.currentStructure, nextMove):
                            self.currentStructure.append(nextMove)
                            del self.possibleRates[i]
                             # remove this rate from happening
                             # remove this stem because now it has been chosen
                            self.possibleRates.insert(0, self.possibleBreakRates[i])
                            #self.possibleRates.append(self.possibleBreakRates[i]) # now append the possiblity of breaking this stem

                            self.possibleRates = kF.normalize(self.possibleRates)
                            #print(kF.calculateTotalFlux(self.possible Rates))
                            print('Time: %0.2fs | Added Stem: %s | Current Structure: %s' %(self.time, str(nextMove), self.convert2dot()))

                            break

                    else:
                    # We have chosen to break the stem so we will remove it from the current structure
                    # self.possibleRates[i][0] will match one of the rates in self.possibleRatesBreak

                        for j in range(len(self.possibleBreakRates)):
                            if self.possibleRates[j][0] == self.possibleBreakRates[j][0]:
                            # found the rate that corresponds with this stem
                                breakStem = self.MemoryOfPossibleStems[j]
                                self.currentStructure = kF.findWhereAndBreak(self.currentStructure, breakStem)
                                del self.possibleRates[i]
                                print('Time: %0.2fs | Broke Stem: %s | Current Structure: %s' %(self.time, str(breakStem), str(self.currentStructure)))
                                self.possibleRates = kF.normalize(self.possibleRates)

        return(self)


    def convert2dot(self):
        # conver to dot bracket notation including pseudoknots
        representation = ''
        dotbracket = [0]*len(self.sequence)
        currentStructure = self.currentStructure[0]
        for i in range(len(currentStructure)):
            open = currentStructure[i][0]
            close = currentStructure[i][1]
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
        while self.time < self.cutoff:
            self.MonteCarloStep()
        return(self.currentStructure)



#'AGGCCAUGGUGCAGCCAAGGAUGACUUGCCGAUCGAUCGAUCUAUCUAUGAAGCUAAGCUAGCUGGCCAUGGAUCCAUCCAUCAAUUGGCAAGUUGUUCUUGGCUACAUCUUGGCCCCU'
#'CGGUCGGAACUCGAUCGGUUGAACUCUAUC'
G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 10)

structure = G.runGillespie()
