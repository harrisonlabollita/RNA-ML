import numpy as np
import RNA
import matplotlib.pyplot as plt
import glob, sys, os
# Finding the ViennaRNA predictions that are very different from the actual structure

def str_append(s):
    output = ''
    output += s
    return output

def readFile(file):
    record = {}
    dots = ''
    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            l = [x for x in line.split() if x != '']
            n = int(l[0])
            base = l[1]
            partner = int(l[4])
            record[n] = [base, partner]
    seq = "".join([record[x][0] for x in sorted(record.keys())])
        # we will take sequences that are between 10 and 500

    if len(seq) > 10 and len(seq) < 500:
        for x in sorted(record.keys()):
            val = record[x][1]
            if val > x:
                s = '('
                dots += str_append(s)
            elif val == 0:
                s = '.'
                dots += str_append(s)
            elif val < x:
                s = ')'
                dots += str_append(s)
    else:
        return(0,0)
    return seq, dots

seq_files = glob.glob('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/data_set/ct_files/*')
#seq_files = os.listdir('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/data_set/ct_files/')

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

sequences = []
dotbrackets = []
seqLengths = []
filenames = []

for file in seq_files:
    head, name = os.path.split(file)

    if len(sequences) < 200:
        seq, dots = readFile(file)

        if seq != 0 and np.random.random() > 0.5:
            filenames.append(name)
            sequences.append(seq)
            dotbrackets.append(dots)
            seqLengths.append(len(seq))
    else:
        break

vPredictions, vEnergies = viennaPredict(sequences)

incorrect = [] # Array to track how many mistakes Vienna RNA made for each sequence, the length will give us the total number of incorrect.
correct = 0 # counter for the number of sequences Vienna RNA correctly predicts
badlyPredictedSequences = []
bpsCorrepsondingStructures = []
bad_incorrect = []

for i in range(len(dotbrackets)):
    # Loop throught all of the source dot-bracket representations
    mistake = 0 # counter for the number of mistakes that Vienna RNA made
    opens = 0
    close = 0
    dot = 0
    for j in range(len(dotbrackets[i])):
        # Run through each sequence of dot-brackets
        val = compare(vPredictions[i][j], dotbrackets[i][j])

        if val == 0:
            mistake +=0
        else:
            mistake += 1
            if val == 1:
                opens +=1
            elif val == 2:
                close +=1
            else:
                dot += 1
    if mistake == 0:
        # if the mistake counter is still 0, then Vienna RNA correctly predcited the base pairs
        correct += 1
    else:
        mistake /= len(dotbrackets[i])
        incorrect.append(mistake)

        # if ViennaRNA missed more than 10% of the predictions, let's keep track of these sequences
        if mistake >= 0.25:
            badlyPredictedSequences.append(sequences[i])
            bpsCorrepsondingStructures.append(dotbrackets[i])
            bad_incorrect.append(mistake)

directory = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/data_set/bad_seq/'

print(len(incorrect))

for i in range(len(badlyPredictedSequences)):
    for j in range(len(sequences)):
        if badlyPredictedSequences[i] == sequences[j]:

            f = open(directory + str(filenames[j])[:-3] + '_%0.2f.ct' %(bad_incorrect[i]), 'w')
            f.write(str(filenames[j])[:-3] + '\n')
            f.write(str(badlyPredictedSequences[i]) + '\n')
            f.write('Actual: ' + str(bpsCorrepsondingStructures[i]) + '\n')
            f.write('RNAfold: ' + str(RNA.fold(badlyPredictedSequences[i])[0]))



print('|---------------------------|')
print('|   Data set Statistics     |')
print('|---------------------------|')
print('|  Avg. Len    |    %d     |' %(np.mean(seqLengths)))
print('|  Shortest    |    %d      |' %(np.min(seqLengths)))
print('|  Longest     |    %d     |' %(np.max(seqLengths)))
print('|---------------------------|')

print('|--------------------------|')
print('|   ViennaRNA Performance  |')
print('|--------------------------|')
print('|  Accuracy   |    %0.2f   |' %((correct/len(dotbrackets))*100))
print('|  Mean       |    %0.2f      |' %(np.mean(incorrect)))
print('|  StDev      |    %0.2f   |' %(np.std(incorrect)))
print('|  Worst      |    %0.2f    |' %(max(incorrect)))
print('|--------------------------|')


############### PLOT ################
#plt.figure()
#plt.title('ViennaRNA Performance')
#plt.grid(True, linestyle = ':', linewidth = 1)
#plt.hist(incorrect, bins = 10, histtype = 'bar', rwidth = 0.9, color = 'black', alpha = 0.9, linewidth = 2)
#plt.xlabel('Errors/Sequence')
#plt.ylabel('Frequency')
#plt.show()
