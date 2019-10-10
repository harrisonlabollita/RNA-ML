import torch
from torch.autograd import Variable
import numpy as np
import time
import model as rnaConvNet
import load_data as load
import plot as p
import accuracy as acc


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
    #totalStep = len(train_loader)
    train_acc = []
    val_acc = []
    losses = []
    val_losses = []

    for epoch in range(Epochs):

        runningLoss = 0.0
        totalTrainLoss = 0.0

        startTime = time.time()

        for i, (src, tgt) in enumerate(train_loader):
            temp_acc_list = []

            src = src.view(batch_size, 1, max_seq_length, max_seq_length)

            src = Variable(src)
            tgt = Variable(tgt)

            outputs = convNet(src)

            loss_size = loss(outputs, tgt)

            # Delete the target and source variables to free up memory
            del src

            optimizer.zero_grad()
            loss_size.backward()
            optimizer.step()

            temp_acc_list.append(np.mean(acc.compute_acc(outputs, tgt)))

            del tgt
            del outputs

        train_acc.append(np.mean(temp_acc_list))

        for i, (pred, real) in enumerate(test_loader):

            temp_val_acc = []

            pred = Variable(pred)
            real = Variable(real)

            pred = pred.view(-1, 1, max_seq_length, max_seq_length)

            val_outputs = convNet(pred)
            val_loss_size = loss(val_outputs, real)

            temp_val_acc.append(np.mean(acc.compute_acc(val_outputs, real)))

            del pred
            del real

        val_acc.append(np.mean(temp_val_acc))
        val_losses.append(float(val_loss_size))
        losses.append(float(loss_size.item()))
        print('Epoch: {}/{}, Accuracy: {:0.2f}%, Loss: {:.4f}, Val loss: {:0.4f}, Val Acc: {:0.2f}%, Time: {:0.2f}s'.format(epoch + 1, Epochs, train_acc[epoch]*100, float(loss_size.item()), float(val_loss_size.item()), val_acc[epoch]*100, time.time() - startTime))

    history = [train_acc, val_acc, losses, val_losses]
    return history

model = rnaConvNet.rnaConvNet(max_seq_length, num_classes)

history = train(model, batch_size, epochs, learning_rate)
torch.save(model.state_dict(), '/Users/harrisonlabollita/Library/Mobile Documents/com~apple~CloudDocs/Arizona State University/Sulc group/src/explore/cnn code/cnn_trained_model.pt')
p.plotmodel_loss(epochs, history[2], history[3], 'Model Loss')
p.plotmodel_acc(epochs, history[0], history[1], 'Model Accuracy')
