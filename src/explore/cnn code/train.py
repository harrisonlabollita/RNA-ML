import torch
from torch.autograd import Variable
import numpy as np
import time
import model as rnaConvNet
import load_data as load
import plot as p


#sources = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/sequences.txt'
#targets = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/dotbrackets.txt'

sources = '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/code/RNA_data_set.csv'
targets = 0

max_seq_length = 30
batch_size = 100
num_classes = 3
epochs = 100
learning_rate = 1e-3



def train(convNet, batch_size, Epochs, learningRate):
    print('-'*30)
    print("  HYPERPARAMETERS  ")
    print('-'*30)
    print("Batch size = ", batch_size)
    print("Epochs = ", Epochs)
    print("Learning rate = ", learningRate)
    print('-'*30)

    # function to call in data
    train_loader, test_loader = load.getTrainingSets(sources, targets, max_seq_length, batch_size)

    # Create loss and optimizer functions
    loss = rnaConvNet.Loss()
    optimizer = rnaConvNet.Optimizer(convNet, learningRate)
    trainingStartTime = time.time()

    # Start training
    totalStep = len(train_loader)
    losses = []
    val_losses = []

    for epoch in range(Epochs):

        runningLoss = 0.0
        totalTrainLoss = 0.0

        startTime = time.time()
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


        for i, (pred, real) in enumerate(test_loader):
            pred = Variable(pred)
            real = Variable(real)

            pred = pred.view(-1, 1, max_seq_length, max_seq_length)

            val_outputs = convNet(pred)
            val_loss_size = loss(val_outputs, real)
            val_losses.append(val_loss_size)

        print('Epoch: {}/{}, Loss: {:.4f}, Val loss: {:0.4f}, Time: {:0.2f}s'.format(epoch + 1, Epochs,  loss_size.item(), val_loss_size.item(), time.time() - startTime))

    return losses, val_losses

model = rnaConvNet.rnaConvNet(max_seq_length, num_classes)

train_loss, validation_loss = train(model, batch_size, epochs, learning_rate)

p.plotmodel(epochs, train_loss, validation_loss, 'CNN Model')
