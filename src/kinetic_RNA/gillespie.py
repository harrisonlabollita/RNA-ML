import numpy as np
import kineticFunctions as kF

############################### ALGORITHMN #####################################



class Gillespie:

    def __init__(self, sequence, frozenBPs, cutoff):
        self.sequence = sequence
        self.frozenBPs = frozenBPs

        self.STableBPs, self.compatibilityMatrix, self.stemEnergies, self.stemEntropies  = self.initialize(sequence, frozenBPs)
        self.time = 0
        self.startingStructure = []
        self.stemsInStructure = []
        self.transitionRates = []
        self.currentStructure = []
        self.totalFlux = 0
        self.cutoff = cutoff


    def initialize(self, sequence, frozenBPs):
    # Run the calculation for the free energy landscape. The calculate free energy
    # landscape function is very thorough. As for now, we are only interested in the
    # following variables:
    #                        numStems           (number of stems)
    #                        numStructures      (number of structures)
    #                        STableStructure    (number of Stems)
    #                        STableBPs          (number of Stems BasePair Format)
    #                        Compatibility Matrix
    #                        Sequence in Numbers

    # Rename variables/information that we will need in our Gillespie algorithmn
        sequenceInNumbers, numStems, STableStructure, STableBPs = kF.createSTable(sequence)
        compatibilityMatrix = kF.makeCompatibilityMatrix(numStems, 1, STableStructure, STableBPs, frozenBPs)
        stemEnergies, stemEntropies = kF.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)

        return(STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)

    def calculateStemRates(self, values, kB, T):
        k_0 = 1.0
        transitionRates = []
        for i in range(len(values)):
            rate = k_0*np.exp(values[i]/(kB*T))
            transitionRates.append(rate)
        return transitionRates

    def calculateTotalFlux(self, rates):
        totalFlux = sum(rates)
        return totalFlux

    def isCompatible(self, stemsInStructure, j, compatibilityMatrix):
    # Idea: could just use the compatibiliy matrix that is already created in
    # RFE.
    # nextMove [[ ntd i, ntd j]]
    # PartiallyFoldedSequence = [[a, b], [c, d], [e, f]....]
    # convert to a list of numbers
        for i in range(len(stemsInStructure)):
            index = stemsInStructure[i]
            if compatibilityMatrix[index, j] == 0:
                return False
        return True


    def MonteCarloStep(self):
    # Following Dykeman 2015 (Kfold) paper

        if len(self.currentStructure) == 0:
            r1 = np.random.random()
            r2 = np.random.random()
            ratesForm = self.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15)
            ratesBreak = self.calculateStemRates(self.stemEnergies, kB = 0.0019872, T = 310.15)

            self.totalFlux = self.calculateTotalFlux(ratesForm)
            self.time -= np.log(r2)/self.totalFlux

            for i in range(len(ratesForm))):

                trial = sum(ratesForm[:i])

                if  trial >= r1*self.totalFlux:
                    print('Forming stem...')
                    nextMove = self.STableBPs[i]
                    self.currentStructure.append(nextMove)
                    self.stemsInStructure.append(i)
                # remove the chosen stem from the list
                    del self.STableBPs[i]
                    del self.ratesFrom[i]

                    for k in range(len(nextMove)):
                        print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                        self.totalFlux = r1*self.totalFlux - sum(ratesForm[:i])
            break

        else:
            r1 = np.random.random()
            r2 = np.random.random()
            self.time -= np.log(r2)/self.totalFlux
            for i in range(len(ratesForm)):
                trial = sum(ratesForm[:i])
                if  trial >= r1*totalFlux:
                    nextMove = allPossibleStems[i]
                    if isCompatible(self.stemsInStructure, i, self.compatibilityMatrix):
                        print('Forming stem...')
                        if nextMove not in self.currentStructure:
                            self.currentStructure.append(nextMove)
                            self.stemsInStructure.append(i)
                        # remove the stem and the rate
                        # idea maybe let's combine these arrays
                            del self.STableBps[i]
                            del self.ratesForm[i]

                            for k in range(len(nextMove)):
                                print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                                self.totalFlux = r1*self.totalFlux - sum(self.ratesForm[:i])
                        break

                    else:
                        inCompatible = []
                    # because our next move was incompatible with the current move, let's break
                    # remove the incompatible one and add the new one
                        for j in range(len(self.stemsInStructure)):
                        # nextMove has index of the ith stem so we use i here
                            if self.compatibilityMatrix[j, i] == 0:
                                inCompatible.append(j)
                        inCompList = [self.STableBPs[m] for m in range(len(inCompatible))]
                        print('Breaking stems...%s' %(str(inCompList)))

                        to_delete = sorted(inCompatible)
                            for d in to_delete:
                                del self.currentStructure[d]
                                del self.stemsInStructure[d]

                        if nextMove not in self.currentStructure:
                            self.currentStructure.append(i) # add the nextMove
                            self.stemsInStructure.append(i)
                            del self.STableBPs[i]
                            del self.ratesForm[i]

                            print('Pairing stems...')
                            for k in range(len(nextMove)):
                                print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                                self.toatlFlux = r1*self.totalFlux - sum(ratesForm)

    return(self)

    while self.time < self.cutoff:
        MonteCarloStep()

Gillespie('AUCUGAUACUGUGCUAUGUCUGAGAUAGC', [], 3)
