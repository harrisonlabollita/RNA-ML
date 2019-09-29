import numpy as np
import RNA 
import torch 


def strAppend(s):
    output = ''
    output += s
    return output
 
def dot2num(array):
    new_array = []
    for elem in array:
	if elem == '(':
	   new_array.append(0)
        elif elem == ')':
           new_array.append(1)
        else:
           new_array.append(2)
    return np.array(new_array)

def num2dot(array):
    dot = ''
    for elem in array:
        if elem == 0:
           dot += strAppend('(')
        elif elem == 1:
           dot += strAppend(')')
        else: 
           dot += strAppend('.')
     return dot


def getRNA(filename):
    # input: file containing RNA sequences 
    # output: array of RNA sequences
    sequences = [] 
    with open(filename, 'r') as f:
	for line in f.readlines():
	    line = line.strip('\n')
            sequences.append(line)
    return np.array(sequences)


def getDotBracket(filename):
    # input: file containing dot bracket representation
    # output: array of dot brackets
    dotbrackets = []
    with open(filename, 'r') as f:
         for line in f.readlines():
	     line = line.strip('\n')
             dotbrackets.append(line)
    return np.array(dotbrackets)

def RNAfoldPredict(sequences):
    # input: array of RNA sequences 
    # output: array of predictions from RNAfold
    for seq in sequences:
        pred = RNA.fold(seq)
        predictions.append(pred)
    return np.array(predictions) 

def TrainingSets(file_seq, file_dots):
    # input: filename1, filename2
    # output: torch datasets 
    sequences = getRNA(file_seq)
    predictions = RNAfoldPredict(sequences)
    dotbrackets = getDotBracket(file_dots)
    # convert the dot bracket representation to number representation for training
    for i in range(len(predicitons)):
        predictions[i] = dot2num(predictions[i])
    for i in range(len(dotbrackets)):
        dotbrackets[i] = dot2num(dotbrackets[i])
    return training_loader, testing_loader
