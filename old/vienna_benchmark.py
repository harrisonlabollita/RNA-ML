import numpy as np
import RNA
import matplotlib.pyplot as plt
import csv

filename = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNA_data_set.csv'
#filename = '/Users/Hank/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNA_data_set.csv'

def readData(filename):
    # Input: filename (as .csv)
    # Output: RNA sequences, dot Bracket representation, and free energy
    with open(filename) as f:
        next(f)
        sequs = csv.reader(f)
        data = []
        for seq in sequs:
            data.append(seq)
        data = np.asarray(data)
        seq, dot, free = refineData(data)
    return seq, dot, free

def refineData(data):
    freeEnergy = []
    sequences = []
    dotBrackets = []
    for i in range(len(data)):
        sequences.append(data[i][0])
        dotBrackets.append(data[i][1])
        freeEnergy.append(data[i][2])
    sequences = np.asarray(sequences)
    dotBrackets = np.asarray(dotBrackets)
    freeEnergy = np.asarray(freeEnergy)
    return sequences, dotBrackets, freeEnergy

def viennaPredict(sequences):
    predictions = []
    energies = []
    for i in range(len(sequences)):
        pred, energy = RNA.fold(sequences[i])
        predictions.append(pred)
        energies.append(energy)

    return predictions, energies


def compare(prediction, target):
    if prediction == target:
        return 0
    else:
        if target == '(':
            return 1
        elif target == ')':
            return 2
        else:
            return 3


sequences, dotBrackets, freeEnergies = readData(filename)

vPredictions, vEnergies = viennaPredict(sequences)


incorrect = [] # Array to track how many mistakes Vienna RNA made for each sequence, the length will give us the total number of incorrect.
correct = 0 # counter for the number of sequences Vienna RNA correctly predicts
openErrors = []
closeErrors = []
dotErrors = []

for i in range(len(dotBrackets)):
    # Loop throught all of the source dot-bracket representations
    mistake = 0 # counter for the number of mistakes that Vienna RNA made
    open = 0
    close = 0
    dot = 0
    for j in range(len(dotBrackets[i])):
        # Run through each sequence of dot-brackets
        val = compare(vPredictions[i][j],dotBrackets[i][j])
        if val == 0:
            mistake +=0
        else:
            mistake += 1
            if val == 1:
                open +=1
            elif val == 2:
                close +=1
            else:
                dot += 1
    if mistake == 0:
        # if the mistake counter is still 0, then Vienna RNA correctly predcited the base pairs
        correct += 1
    else:
        incorrect.append(mistake)
        openErrors.append(open)
        closeErrors.append(close)
        dotErrors.append(dot)

totalErrors = np.sum(openErrors) + np.sum(closeErrors) + np.sum(dotErrors)
lessThan = 0
for i in range(len(incorrect)):
    if incorrect[i] < 10:
        lessThan+=1

print('|--------------------------|')
print('|   ViennaRNA Performance  |')
print('|--------------------------|')
print('|  Accuracy   |    %0.2f   |' %((correct/len(dotBrackets))*100))
print('|  Mean       |    %d       |' %(np.mean(incorrect)))
print('|  StDev      |    %0.2f    |' %(np.std(incorrect)))
print('|  Worst      |    %0.2f   |' %(max(incorrect)))
print('|  openErr    |    %0.2f   |' %((np.sum(openErrors)/totalErrors)*100))
print('|  closeErr   |    %0.2f   |' %((np.sum(closeErrors)/totalErrors)*100))
print('|  dotErr     |    %0.2f   |'  %((np.sum(dotErrors)/totalErrors)*100))
print('|--------------------------|')




plt.figure()
plt.title('ViennaRNA Performance')
plt.grid(True, linestyle = ':', linewidth = 1)
plt.hist(incorrect, bins = 10, histtype = 'bar', rwidth = 0.9, color = 'black', alpha = 0.9, linewidth = 2)
plt.text(20, 17500, r'Mean: %d' %(np.mean(incorrect)))
plt.text(20, 16700, r'StDev: %0.2f' %(np.std(incorrect)))
plt.text(20, 15900, r'< 10 errs: %d' %(lessThan))
plt.xlabel('Errors/Sequence')
plt.ylabel('Frequency')
plt.show()
