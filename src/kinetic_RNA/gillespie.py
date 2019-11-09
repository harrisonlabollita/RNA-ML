import numpy as np
import kineticFunctions as kF


############################### ALGORITHMN #####################################

class Gillespie:

    def __init__(self, sequence, frozenBPs, cutoff):
        self.sequence = sequence
        self.frozenBPs = frozenBPs
        self.STableBPs, self.compatibilityMatrix, self.stemEnergies, self.stemEntropies  = self.initialize(sequence, frozenBPs)
        self.startingStructure = []
        self.stemsInStructure = []
        self.transitionRates = []
        self.currentStructure = []
        self.totalFlux = 0
        self.cutoff = cutoff
        self.time = 0
        self.makeOutputFile = False

        if self.makeOutputFile:
            self.f = open('output.txt', 'w+')
            self.f.write('Sequence: %s\n' %(self.sequence))


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
        frozenStems = kF.frozenStemsFromFrozenBPs(frozenBPs, STableBPs, numStems)
        compatibilityMatrix = kF.makeCompatibilityMatrix(numStems, 1, STableStructure, STableBPs, frozenStems)
        stemEnergies, stemEntropies = kF.calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
        return(STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)


    def flatten(self, x):
        out = []
        for i in range(len(x)):
            out.append(x[i][0])
            out.append(x[i][1])
        return out

    def calculateStemRates(self, values, kB, T, kind):
        k_0 = 1.0
        transitionRates = []
        if kind:
            # then we are calculating the rates of forming stems
            for i in range(len(values)):
                rate = [k_0*np.exp(values[i]/(kB*T)), 1]
                transitionRates.append(rate)
        else:

            # then we are calculating the rates of breaking stems
            for i in range(len(values)):
                rate = [k_0*np.exp(values[i]/(kB*T)), 0]
                transitionRates.append(rate)

        return transitionRates

    def calculateTotalFlux(self, rates):
        totalFlux = 0
        for i in range(len(rates)):
            totalFlux += rates[i][0]
        return totalFlux

    def isCompatible(self, stemsInStructure, j, compatibilityMatrix):
    # Idea: could just use the compatibiliy matrix that is already created in
    # RFE.
    # nextMove [[ ntd i, ntd j]]
    # PartiallyFoldedSequence = [[a, b], [c, d], [e, f]....]
    # convert to a list of numbers
        com = []
        for i in range(len(stemsInStructure)):
            index = stemsInStructure[i]
            if compatibilityMatrix[index, j] == 0:
                com.append(j)
        return com

    def partialSum(self, rates):
        partial = 0
        for i in range(len(rates)):
            partial += rates[i][0]
        return partial

    def canAdd(self, stems, new_stem):
        s = np.ravel(stems)
        n = np.ravel(new_stem)
        for i in n:
            if i in s:
                return False
        return True

    def MonteCarloStep(self):

    # Following Dykeman 2015 (Kfold) paper

        if len(self.currentStructure) == 0:
            self.MemoryOfStems = self.STableBPs # this will be used for when we need to access a stem that we are breaking

            r1 = np.random.random()
            r2 = np.random.random()
            self.rates = self.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15, kind = 1)
            self.ratesBreak = self.calculateStemRates(self.stemEnergies, kB = 0.0019872, T = 310.15, kind = 0)
            self.totalFlux = self.calculateTotalFlux(self.rates)

            self.time = (-1)*np.log(r2)/self.totalFlux

            for i in range(len(self.rates)):

                trial = self.partialSum(self.rates[:i])

                if  trial >= r1*self.totalFlux:
                    # at this point we can only form a stem so we do not need to check whether this is a forming move or a breaking move
                    nextMove = self.STableBPs[i]
                    self.currentStructure.append(nextMove)
                    self.stemsInStructure.append(i)
                    self.rates.append(self.ratesBreak[i]) # append the rate of breaking for the stem that we have just added. Now as part of our ensemble of
                                                          # of possible moves, we could choose to break this stem.

                # remove the chosen stem from the list
                    del self.STableBPs[i]
                    del self.rates[i]

                    if self.makeOutputFile:
                        self.f.write('Forming stems....\n')
                        for k in range(len(nextMove)):
                            self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))
                    self.totalFlux = r1*self.totalFlux - self.partialSum(self.rates[:i]) # recalculate the totaFlux
                    break
        else:
            # we will always be in this part of the code after we have made our first stem
            r1 = np.random.random()
            r2 = np.random.random()

            self.time = self.time + (np.log(r2)/self.totalFlux)

            for i in range(len(self.rates)):

                trial = sum(self.rates[:i])

                if  trial >= r1*self.totalFlux:
                    if i >= len(self.STableBPs):
                        break
                    # note: if rates[i][1] == 1 this is a forming rate
                    #       if rates[i][1] == 0 this is a breaking rate

                    if self.rates[i][1]:
                        nextMove = self.STableBPs[i]
                        if len(self.isCompatible(self.stemsInStructure, i, self.compatibilityMatrix)) == 0:
                            if self.canAdd(self.currentStructure, nextMove) and i not in self.stemsInStructure:
                                self.currentStructure.append(nextMove)
                                self.stemsInStructure.append(i)
                                self.rates.append(self.ratesBreak[i]) # we will now append the opportunity to break this stem in the future
                                # remove the stem and the rate
                                del self.STableBPs[i] # remove this chosen stem from the possible formed stems
                                del self.ratesForm[i]

                            if self.makeOutputFile:
                                self.f.write('Forming stems...\n')
                                for k in range(len(nextMove)):
                                    self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))
                            self.totalFlux = r1*self.totalFlux - self.partialSum(self.rates[:i])

                        else:
                            break
                        # The next move is not compatible with the the current folded structure. So we will need to break the incompatible parts
                        # of the structure

                            #inCompatible = self.isCompatible(self.stemsInStructure, i, self.compatibilityMatrix) # finds all of the incompatible stems from the compatibility matrix
                            #inCompList = sorted([self.STableBPs[m] for m in range(len(inCompatible))]) # sort the list in such a way so that we can remove the incompatible elements

                            #if len(inCompList) < len(self.currentStructure):
                                # if we need to break more stems than have formed then this is not a good move at all.
                             #check to make sure we are allowed to break the stems
                            # if self.makeOutputFile:
                            #    self.f.write('Breaking stems...%s\n' %(str(inCompList)))
                            #for d in range(len(inCompList)):
                            #    del self.currentStructure[d]n
                            #    del self.stemsInStructure[d]

                            #if self.canAdd(self.currentStructure, nextMove) and i not in self.stemsInStructure:

                            #    self.currentStructure.append(nextMove) # add the next move to the current structure
                            #    self.stemsInStructure.append(i)
                            #    del self.STableBPs[i]
                            #    del self.ratesForm[i]
                            #    if self.makeOutputFile:
                            #        self.f.write('Forming stems...\n')
                            #        for k in range(len(nextMove)):
                            #            self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))
                            #    self.totalFlux = r1*self.totalFlux - sum(self.rates)
                            #    break
                else:
                    # we have now choosen to break this stem
                    # if we have choosen to break the stem then we will remove it from our current stucture and remove this move from our ensemble of moves
                    # rates[i][1] == 0
                    # this corresponds to choosing to break the i'th stem which is in our current structure
                    for j in range(len(self.ratesBreak)):
                        if rates[i][0] == self.ratesBreak[j]:
                            breakThisStem = self.MemoryOfStem[j]
                            # then we have found the rate that matches the j'th stem that we need to remove from our current Structure:
                            for k in range(len(self.currentStructure)):
                                if self.currentStructure[j] == breakThisStem:
                                    del self.currentStructure[j]
                                    break
                                else:
                                    print('Error: Move chosen to break stem %s, but can not find this stem in the current structure!' %(str(breakThisStem)))
                            self.totalFlux = r1*self.totalFlux - self.partialSum(self.rates[:i])
                            break
        return(self)

    def runGillespie(self):
        self.MonteCarloStep()
        while self.time < self.cutoff:
            self.MonteCarloStep()
        return(self.currentStructure)

    def avgRunGillespie(self, N):
        # N - number of trials
        # find the output of the structure and keep track of each output and the frequency of these output
        i = 0
        arrayOfOutputs = []

        while i < N:
            output = self.flatten(self.runGillespie()[0])
            arrayOfOutputs.append(output)
            i +=1

        # now find the number of times each output occured in our sampling process
        frequencyOfOutputs = []
        uniqueOutputs = []
        for j in range(len(arrayOfOutputs)):
            out = arrayOfOutputs[j]
            if len(uniqueOutputs) == 0:
                uniqueOutputs.append(out)
                frequencyOfOutputs.append(arrayOfOutputs.count(out))
            else:
                notFound = 0
                for k in range(len(uniqueOutputs)):
                    if out == uniqueOutputs[k]:
                        notFound = 1
                if notFound == 0:
                    uniqueOutputs.append(out)
                    frequencyOfOutputs.append(arrayOfOutputs.count(out))

        return uniqueOutputs, frequencyOfOutputs

############################## test sequences ##################################
# CGGUCGGAACUCGAUCGGUUGAACUCUAUC  (((((((...)))))))............. [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10]]
# GUUAGCACAUCGAGCGGGCAAUAUGUACAU  (((.((.......)).)))........... [[0, 18], [1, 17], [2, 16], [4, 14], [5, 13] ]
# GAUGCGCAAAAACAUUCCCUCAUCACAAUU  ((((................))))...... [[0, 23], [1, 22], [2, 21], [3, 20]]
# add a functionality to take in multiple sequences as once

#sequences = ['CGGUCGGAACUCGAUCGGUUGAACUCUAUC', 'GUUAGCACAUCGAGCGGGCAAUAUGUACAU', 'GAUGCGCAAAAACAUUCCCUCAUCACAAUU']
#for seq in sequences:
#    G = Gillespie(seq, [18, 19], 2)
#    structure = G.runGillespie()
#    print(structure)

G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 2)

#G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 2)
structure = G.runGillespie()
print('Sequence:' , G.sequence)
print('Structure:', structure)

# Average Gillespie
#outputs, frequencies = G.avgRunGillespie(10)
#print(outputs)
#print(frequencies)
