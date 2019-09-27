# load_data.py
# Input: sequences text file and dotbrackets text file, maximum sequence length
# Ouput: Numpy array of sequnces that are less than or equal to the max. Pads the sequences that are shorter than the maximum length


import numpy as np
import rna2matrix as matrix
import torch

def getRNA(filename, max_seq_length):
    sequences = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) <= max_seq_length:
                seq = []
                diff = int(max_seq_length - len(line))
                for char in line:
                    seq.append(str(char))
                for i in range(len(seq), len(seq) + diff):
                    seq.append(0)
                sequences.append(seq)
    return sequences

def getDotBrackets(filename, max_seq_length):
    bracket_reps = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip('\n')
            if len(line) <= max_seq_length:
                rep = []
                diff = int(max_seq_length - len(line))
                for char in line:
                    if char == '(':
                        rep.append([1, 0, 0])
                    elif char == ')':
                        rep.append([0, 1, 0])
                    else:
                        rep.append([0, 0, 1])
                for i in range(len(rep), len(rep) + diff):
                    rep.append([0, 0, 0])
                rep = np.array(rep).reshape(max_seq_length, 3)
                bracket_reps.append(rep)
    return np.array(bracket_reps)


def getTrainingSets(sources, targets, max_seq_length, batch_size):

    seqs = getRNA(sources, max_seq_length)
    dots = getDotBrackets(targets, max_seq_length)

    if len(seqs) != len(dots):
        raise 'Source data and target data must be the same length!'
    else:

        seqs = matrix.seqs2matrices(seqs)
        N = int(np.ceil(0.8*len(seqs)))
        training_source = torch.FloatTensor(seqs[:N])
        training_source = training_source.view(len(training_source), 1, max_seq_length, max_seq_length)
        training_target = torch.FloatTensor(dots[:N])
        valid_source = torch.FloatTensor(seqs[N:])
        valid_source = valid_source.view(len(valid_source), 1, max_seq_length, max_seq_length)
        valid_target = torch.FloatTensor(seqs[N:])

        train_set = [training_source, training_target]
        test_set = [valid_source, valid_target]


        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
