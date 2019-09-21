import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

def Loss(net, learning_rate):
    loss = nn.CrossEntropyLoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)


def train(convNet, batchSize, Epochs, LearningRate):

    # Fancy intro to net
    print("HYPERPARAMETERS")
    print("batchSize = ", batchSize)
    print("Epochs = ", Epochs)
    print("LearningRate = ", LearningRate)

    # function to call in data

    # Create loss and optimizer functions

    loss = Loss(net, learningRate)
    optimizer = Optimizer(net, learningRate)

    trainingStartTime = time.time()

    # Start training
    totalStep = len(trainLoader)
    losses = []
    accuracies = []

    for epoch in range(Epochs):

        runningLoss = 0.0
        startTime = time.time()
        totalTrainLoss = 0

        for i, data in enumerate(trainLoader, 0):
            inputs, targets = data

            optimizer.zero_grad()

            outpus = convNet(inputs)
            lossSize = loss(outputs, targets)
            losses.append(loss.item())
            lossSize.backward()
            optimizer.step()

            total = targets.size()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            accuracies.append(correct/total)

            runningLoss += lossSize.data[0]
            totalTrainLoss += lossSize.data[0]

            if (i+1) %(100) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch + 1, Epochs, i+1, totalStep, loss.item(),
                        (correct/total)*100))

            totalValLoss = 0
            for inputs, targets in valLoader:
                valOutputs = convNet(inputs)
                valLossSize = loss(valOutputs, targest)
                totalValLoss += valLossSize.data[0]
            print('Validation Loss: {:.2f}'.format(totalValLoss/ len(valLoader)))


        print('Training finished in {:.2f}s'.format(time.time() - trainingStartTime))
