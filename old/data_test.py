import numpy as np
import sys
sys.path.append('/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/cnn code/')
import prepare_data as data

source = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
target = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'

sequences = data.RNAsequences(source)
dot_brackets = np.array(data.structureRep(target))

matrices = np.array(data.seqs2matrices(sequences))

condition = True
while condition:
    if matrices.shape != (len(matrices), 30, 30):
        condition = False
        print('There is an error in processing the data!')
    elif dot_brackets.shape != (len(dot_brackets), 30, 3):
        condition = False
        print('There is an error in processing the data!')
    elif dot_brackets.reshape(len(dot_brackets), 30*3 ).shape != (len(dot_brackets), 90):
        condition = False
        print('There is an error in processing the data!')
    else:
        break
print('All clear!')
