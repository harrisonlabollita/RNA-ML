import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time



def train(convNet, batch_size, Epochs, learningRate):
    print('-'*20)
    print("HYPERPARAMETERS")
    print('-'*20)
    print("Batch size = ", batch_size)
    print("Epochs = ", Epochs)
    print("Learning rate = ", learningRate)
    print('-'*20)
