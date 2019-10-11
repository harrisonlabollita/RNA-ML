import torch
import numpy as np
import rna2matrix as matrix
import model as rnaConvNet
import load_data as load
import nussinov as nussinov
import RNA

def getFile(filename):
    array = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip()
            array.append(line)
    return np.array(array)

def out2dot(matrix):
    out = ''
    for i in range(len(matrix)):
        val, index = torch.max(matrix[i], 1)
        for i in range(len(index)):

            if index[i] == 0:
                out += '('
            elif index[i] == 1:
                out += ')'
            else:
                out += '.'
    return out

def testModel(seq, tgt):
    accuracies = []
    if len(seq) != len(tgt):
        print('Length of files do not match!')

    for i in range(len(seq)):
        test = matrix.RNAmatrix(seq[i])
        test = torch.from_numpy(test)
        test = test.view(1, 1, 30, 30)
        output = model(test.type('torch.FloatTensor'))
        output = out2dot(output)
        accuracies.append(modelAccuracy(output, tgt[i]))

        if i % 500 == 0:
            print('RNA Fold:', RNA.fold(seq[i])[0])
            print('Nussinov:', nussinov.nussinov(seq[i]))
            print('CNN:', output)
            print('Actual:', tgt[i])
            print('                ')
    return np.array(accuracies)

def modelAccuracy(pred, tgt):
    count = 0
    for i in range(len(pred)):
        if pred[i] == tgt[i]:
            count +=1
    count /= len(pred)
    return count

PATH = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/cnn code/cnn_trained_model.pt'

test_seq = getFile('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt')
test_tgt = getFile('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt')

model = rnaConvNet.rnaConvNet(30, 3)
model.load_state_dict(torch.load(PATH))
model.eval()

accuracies = testModel(test_seq, test_tgt)

print('Model Accuracy:', np.mean(accuracies))
print('Best Prediction:', np.max(accuracies))
print('Worst Prediction:', np.min(accuracies))
