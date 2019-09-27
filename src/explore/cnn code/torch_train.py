import torch
import torch.optim as optim
import numpy as np
import time
import torch_model as rnaConvNet
import load_data as load

sources = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt'
targets = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/dotbrackets.txt'


def Loss():
    loss = torch.nn.CrossEntropyLoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)

def train(convNet, batch_size, Epochs, learningRate):

    # Fancy intro to net
    print("HYPERPARAMETERS")
    print("batchSize = ", batch_size)
    print("Epochs = ", Epochs)
    print("LearningRate = ", learningRate)

    # function to call in data
    train_loader, test_loader = load.getTrainingSets(sources, targets, 30, batch_size)

    # Create loss and optimizer functions
    loss = Loss()
    optimizer = Optimizer(convNet, learningRate)
    trainingStartTime = time.time()

    # Start training
    totalStep = len(train_loader)


    losses = []
    accuracies = []

    for epoch in range(Epochs):
        runningLoss = 0.0
        startTime = time.time()
        totalTrainLoss = 0

        for i, (src, tgt) in enumerate(train_loader):

            outputs = convNet(srcs)
            lossSize = loss(outputs, tgts)
            losses.append(loss.item())
            optimizer.zero_grad()
            lossSize.backward()
            optimizer.step()

            total = tgt.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == tgts).sum().item()
            accuracies.append(correct/total)

            runningLoss += lossSize.data[0]
            totalTrainLoss += lossSize.data[0]

            if (i+1) %(100) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                            .format(epoch + 1, Epochs, i+1, totalStep, loss.item(),
                            (correct/total)*100))

        totalValLoss = 0

        for j in range(len(test_src)):
            tst_in = test_src[j]
            test_out  = test_tgt[j]
            valOutputs = convNet(tst_in)
            valLossSize = loss(valOutputs, tst_out)
            totalValLoss += valLossSize.data[0]
            print('Validation Loss: {:.2f}'.format(time.time() - trainingStartTime))



        print('Training finished in {:.2f}s'.format(time.time() - trainingStartTime))

max_seq_length = 30
batch_size = 100
num_class = 3
model =rnaConvNet.rnaConvNet(max_seq_length, num_class, batch_size)
train(model, batch_size, 10, 0.1)
