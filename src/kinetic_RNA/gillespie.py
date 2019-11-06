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
        com = []
        for i in range(len(stemsInStructure)):
            index = stemsInStructure[i]
            if compatibilityMatrix[index, j] == 0:
                com.append(j)
        return com


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

            r1 = np.random.random()
            r2 = np.random.random()
            self.ratesForm = self.calculateStemRates(self.stemEntropies, kB =  0.0019872, T = 310.15)
            #self.ratesBreak = self.calculateStemRates(self.stemEnergies, kB = 0.0019872, T = 310.15)

            self.totalFlux = self.calculateTotalFlux(self.ratesForm)
            self.time = (-1)*np.log(r2)/self.totalFlux
            for i in range(len(self.ratesForm)):
                trial = sum(self.ratesForm[:i])

                if  trial >= r1*self.totalFlux:
                    nextMove = self.STableBPs[i]

                    self.currentStructure.append(nextMove)
                    self.stemsInStructure.append(i)
                # remove the chosen stem from the list
                    del self.STableBPs[i]
                    del self.ratesForm[i]

                    if self.makeOutputFile:
                        self.f.write('Forming stems....\n')
                        for k in range(len(nextMove)):
                            self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))
                    self.totalFlux = r1*self.totalFlux - sum(self.ratesForm[:i]) # recalculate the flux
                    break

        else:

            r1 = np.random.random()
            r2 = np.random.random()

            self.time = self.time + (np.log(r2)/self.totalFlux)

            for i in range(len(self.ratesForm)):
                trial = sum(self.ratesForm[:i])
                if  trial >= r1*self.totalFlux:

                    if i >= len(self.STableBPs):
                        break

                    nextMove = self.STableBPs[i]
                    if len(self.isCompatible(self.stemsInStructure, i, self.compatibilityMatrix)) == 0:
                        if self.canAdd(self.currentStructure, nextMove) and i not in self.stemsInStructure:
                            self.currentStructure.append(nextMove)
                            self.stemsInStructure.append(i)
                        # remove the stem and the rate
                            del self.STableBPs[i]
                            del self.ratesForm[i]
                            if self.makeOutputFile:
                                self.f.write('Forming stems...\n')
                                for k in range(len(nextMove)):
                                    self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))


                            self.totalFlux = r1*self.totalFlux - sum(self.ratesForm[:i])
                    else:
                        # The next move is not compatible with the the current folded structure. So we will need to break the incompatible parts
                        # of the structure

                        inCompatible = self.isCompatible(self.stemsInStructure, i, self.compatibilityMatrix) # finds all of the incompatible stems from the compatibility matrix

                        inCompList = sorted([self.STableBPs[m] for m in range(len(inCompatible))]) # sort the list in such a way so that we can remove the incompatible elements

                        if len(inCompList) < len(self.currentStructure):
                             # if we need to break more stems than have formed then this is not a good move at all.
                             #check to make sure we are allowed to break the stems
                            if self.makeOutputFile:
                                self.f.write('Breaking stems...%s\n' %(str(inCompList)))
                            for d in range(len(inCompList)):
                                del self.currentStructure[d]
                                del self.stemsInStructure[d]

                            if self.canAdd(self.currentStructure, nextMove) and i not in self.stemsInStructure:

                                self.currentStructure.append(nextMove) # add the next move to the current structure
                                self.stemsInStructure.append(i)
                                del self.STableBPs[i]
                                del self.ratesForm[i]
                                if self.makeOutputFile:
                                    self.f.write('Forming stems...\n')
                                    for k in range(len(nextMove)):
                                        self.f.write('Pair: %s - %s\n' %(str(nextMove[k][0]), str(nextMove[k][1])))
                                self.totalFlux = r1*self.totalFlux - sum(self.ratesForm)
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

G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [[0, 16], [1,15]], 2)

#G = Gillespie('CGGUCGGAACUCGAUCGGUUGAACUCUAUC', [], 2)
structure = G.runGillespie()
print('Sequence:' , G.sequence)
print('Structure:', structure)

# Average Gillespie
#outputs, frequencies = G.avgRunGillespie(10)
#print(outputs)
#print(frequencies)
