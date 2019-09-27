import numpy as np
import matplotlib.pyplot as plt
import importData as data
import time

#filename_train_src = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtrain_src.txt'
#filename_train_tgt = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtrain_tgt.txt'
#filename_test_src = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtest_src.txt'
#filename_test_tgt = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNAtest_tgt.txt'


filename_train_src = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
filename_train_tgt = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'
filename_test_src = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_src.txt'
filename_test_tgt = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/test_tgt.txt'


train_src = np.array(data.getRNA(filename_train_src))
train_tgt = np.array(data.getDotBrackets(filename_train_tgt))
train_tgt = train_tgt.reshape(len(train_tgt), 30, 3)
test_src = np.array(data.getRNA(filename_train_src))
test_tgt = np.array(data.getDotBrackets(filename_train_tgt))
test_tgt = test_tgt.reshape( len(test_tgt), 30, 3)


start = time.time()
print('Converting training sequences to matrices')
train_src = data.seqs2matrices(train_src)
print(train_src.shape)
print('Done!')
print('Converting testing sequences to matrices')
test_src = data.seqs2matrices(test_src)
print(test_src.shape)
print('Done!')
end = time.time()
print('Sequence to matrices total time: %0.3f'%(end - start))



import keras_model as cnn
train_src = train_src.reshape(len(train_src), 30, 30, 1)
test_src = test_src.reshape(len(test_src), 30, 30, 1)


model, history = cnn.Model(train_src, train_tgt, test_src, test_tgt)
# model.save('RNA_CNN_Model.h5')
score = model.evaluate(test_src, test_tgt, verbose = 0)
print('Model Performance: Loss %.2f' %(score[0]))
print('Model Performance: Accuracy %.5f' %(score[1]))

pred = model.predict(test_src[0].reshape(1, 30, 30, 1))
print(pred)
