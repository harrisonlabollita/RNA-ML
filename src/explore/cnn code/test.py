import torch
import numpy as np
import rna2matrix as matrix
import model as rnaConvNet
import load_data as load
import nussinov as nussinov
import RNA

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

PATH = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/cnn code/cnn_trained_model.pt'

test_seq = 'CGGUCGGAACUCGAUCGGUUGAACUCUAUC'
test_tgt = '(((((((...))))))).............'

model = rnaConvNet.rnaConvNet(30, 3)
model.load_state_dict(torch.load(PATH))
model.eval()

test = matrix.RNAmatrix(test_seq)
test = torch.from_numpy(test)
test = test.view(1, 1, 30, 30)
output = model(test.type('torch.FloatTensor'))
output = out2dot(output)

print('RNA Fold:', RNA.fold(test_seq)[0])
print('Nussinov:', nussinov.nussinov(test_seq))
print('CNN:', output)
print('Real:', test_tgt)
