############################### ALGORITHMN #####################################

def initialize(sequence):
    # Run the calculation for the free energy landscape. The calculate free energy
    # landscape function is very thorough. As for now, we are only interested in the
    # following variables:
    #                        numStems           (number of stems)
    #                        numStructures      (number of structures)
    #                        STableStructure    (number of Stems)
    #                        STableBPs          (number of Stems BasePair Format)

    R = RNALandscape([sequence], storeGraphs = False, makeFigures = False,
    printProgressUpdate = False, toSave = False, tryToLoadVariables = False)
    R.calculateFELandscape()
    # Rename variables/information that we will need in our Gillespie algorithmn
    numStems = R.numStems
    numStructures = R.numStructures
    STableStructure = R.STableStructure
    STableBPs = R.STableBPs
    indexSort = R.indexSort
    sortedProbs = R.sortedProbs
    sortedFEs = R.sortedFEs
    compatibilityMatrix = R.C
    sequenceInNumbers = R.sequenceInNumbers
    stemEnergies, stemEntropies = calculateStemFreeEnergiesPairwise(numStems, STableStructure, sequenceInNumbers)
    return(sequenceInNumbers, numStems, numStructures, STableStructure, STableBPs, compatibilityMatrix, stemEnergies, stemEntropies)

sequence = 'AUCUGAUACUGUGCUAUGUCUGAGAUAGC'

sequenceInNumbers, numStems, numStructures, STableStructure, STableBPs, compatibilityMatrix, stemEnergies, stemEntropies = initialize(sequence)

def calculateStemTransitionRates(stemEntropies, kB, T):
    k_0 = 1.0
    transitionRates = []
    for i in range(len(stemEntropies)):
        rate = k_0*np.exp(stemEntropies[i]/(kB*T))
        transitionRates.append(rate)
    return transitionRates


def calculateTotalFlux(rates):
    totalFlux = sum(rates)
    return totalFlux


def isCompatible(stemsInStructure, j, compatibilityMatrix):
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


def MonteCarloStep(currentStructure, stemsInStructure, allPossibleStems, compatibilityMatrix, time, stemEntropies, totalFlux, transitionRates):
    # Following Dykeman 2015 (Kfold) paper
    if len(currentStructure) == 0:
        r1 = np.random.random()
        r2 = np.random.random()
        transitionRates = calculateStemTransitionRates(stemEntropies, kB =  0.0019872, T = 310.15)
        totalFlux = calculateTotalFlux(transitionRates)
        time -= np.log(r2)/totalFlux
        for i in range(len(transitionRates)):
            trial = sum(transitionRates[:i])
            if  trial >= r1*totalFlux:
                print('Forming stem...')
                nextMove = allPossibleStems[i]
                currentStructure.append(nextMove)
                stemsInStructure.append(i)
                # remove the chosen stem from the list
                del allPossibleStems[i]
                del transitionRates[i]

                for k in range(len(nextMove)):
                    print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                totalFlux = r1*totalFlux - sum(transitionRates[:i])
                break


    else:
        r1 = np.random.random()
        r2 = np.random.random()
        totalFlux = totalFlux
        time -= np.log(r2)/totalFlux
        for i in range(len(transitionRates)):
            trial = sum(transitionRates[:i])
            if  trial >= r1*totalFlux:
                nextMove = allPossibleStems[i]
                if isCompatible(stemsInStructure, i, compatibilityMatrix):
                    print('Forming stem...')
                    if nextMove not in currentStructure:
                        currentStructure.append(nextMove)
                        stemsInStructure.append(i)
                        # remove the stem and the rate
                        # idea maybe let's combine these arrays
                        del allPossibleStems[i]
                        del transitionRates[i]

                        for k in range(len(nextMove)):
                            print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                        totalFlux = r1*totalFlux - sum(transitionRates[:i])
                        break
                else:

                    inCompatible = []
                    # because our next move was incompatible with the current move, let's break
                    # remove the incompatible one and add the new one
                    for j in range(len(stemsInStructure)):
                        # nextMove has index of the ith stem so we use i here
                        if compatibilityMatrix[j, i] == 0:
                            inCompatible.append(j)
                    inCompList = [allPossibleStems[m] for m in range(len(inCompatible))]
                    print('Breaking stems...%s' %(str(inCompList)))

                    to_delete = sorted(inCompatible)
                    for d in to_delete:
                        del currentStructure[d]
                        del stemsInStructure[d]

                    if nextMove not in currentStructure:
                        currentStructure.append(i) # add the nextMove
                        stemsInStructure.append(i)
                        del allPossibleStems[i]
                        del transitionRates[i]

                    print('Pairing stems...')
                    for k in range(len(nextMove)):
                        print('Pair: %s - %s' %(str(nextMove[k][0]), str(nextMove[k][1])))
                    toatlFlux = r1*totalFlux - sum(transitionRates)



    # Need a terminating condition:
    # How much time can the folding take?
    # or until structure is created?

    return currentStructure, stemsInStructure, allPossibleStems, totalFlux, time, transitionRates

startingStructure = []
stemsInStructure = []
transRates = []
cS, stemS, possStems, t, tflux, transRates = MonteCarloStep(startingStructure, stemsInStructure, STableBPs, compatibilityMatrix, 0, stemEntropies, 0, transRates)

start = 0
max = 3
while start < max:
    cS, stemS, possStems, t, tflux, transRates = MonteCarloStep(cS, stemS, possStems, compatibilityMatrix, t, stemEntropies, tflux, transRates)
    start += 1
