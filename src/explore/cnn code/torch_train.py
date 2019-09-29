import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import torch_model as rnaConvNet
import load_data as load


#sources = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt'
#targets = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/dotbrackets.txt'

sources = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNA_data_set.csv'
targets = 0
max_seq_length = 30
batch_size = 100
num_classes = 3
epochs = 100
learning_rate = 1e-4


def Loss():
    loss = torch.nn.BCEWithLogitsLoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)
    return optimizer

def train(convNet, batch_size, Epochs, learningRate):
    print('-'*20)
    print("HYPERPARAMETERS")
    print('-'*20)
    print("Batch size = ", batch_size)
    print("Epochs = ", Epochs)
    print("Learning rate = ", learningRate)
    print('-'*20)

    # function to call in data
    train_loader, test_loader = load.getTrainingSets(sources, targets, max_seq_length, batch_size)
    print('Load data successful!')

    # Create loss and optimizer functions
    loss = Loss()
    optimizer = Optimizer(convNet, learningRate)
    trainingStartTime = time.time()

    # Start training
    totalStep = len(train_loader)
    losses = []

    for epoch in range(Epochs):


        runningLoss = 0.0
        startTime = time.time()
        totalTrainLoss = 0

        for i, (src, tgt) in enumerate(train_loader):

            src = src.view(batch_size, 1, max_seq_length, max_seq_length)

            src = Variable(src)
            tgt = Variable(tgt)

            outputs = convNet(src)
            loss_size = loss(outputs, tgt)
            losses.append(loss_size.item())

            optimizer.zero_grad()
            loss_size.backward()
            optimizer.step()

            runningLoss += loss_size.item()
            totalTrainLoss += loss_size.item()

        print('Epoch [{}/{}], Loss: {:.4f}, Time: {:0.2f}s'.format(epoch + 1, Epochs,  loss_size.item(), time.time() - startTime))



#        totalValLoss=0
#
#        for test, out in test_loader:
#            test = test.view(batch_size, 1, max_seq_length, max_seq_length)
#
#            valOutputs = convNet(test)
#            valLossSize = loss(valOutputs, out)
#            totalValLoss += valLossSize.item()
#
#            print('Validation Loss: {:.2f}'.format(totalValLoss /len(test_loader)))
    return losses

model = rnaConvNet.rnaConvNet(max_seq_length, num_classes, batch_size)
train(model, batch_size, epochs, learning_rate)
