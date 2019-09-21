# CONVOLUTIONAL NEURAL NETWORK IN PYTORCH
# Author: Harrison LaBollita
# Version: 1.0


# Model framework
# Input Layer: Matrix Representation of Bases
# Hidden Layer 1: Convolution (16) (3,3)
# Pooling (3,3)
# Hidden Layer 2: Convolution (16) (3, 3)
# Pooling (3,3)
# Fully Connected Layer 1
# Fully Connected Layer 2

# Input Layer Dimensions: (batchSize, 1, 30, 30)
# Convolution Layer 1: (in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1)
#             Output Shape: (batchSize, 16, 27, 27)
# Convolution Layer 2: (in_channels = 15, out_channels = 16, kernel_size = 3, stride = 1)
#             Output Shape: (batchSize, 16, 24, 24)
# There are batchSize * 16 * 24 * 24 parameters so reshape to (3072, 30)
# Hidden Layer 1: (3072, 30)
#             OutputShape: (4096, 72000)
# Hidden Layer 2: (30, 30)
#             OutputShape: (4096, 4096)
# Hidden Layer 3: (30, 3)
#              OutputShape: (4096, 3)


import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F



class rnaConvNet(torch.nn.Module):

    def __init__(self):

        super(rnaConvNet, self).__init__()

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =1)
            )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size = 3, stride =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1)
            )
        self.fullyConnect1 = nn.Sequential(
            nn.Linear(3072, 30),
            nn.ReLU()
            )
        self.fullyConnect2 = nn.Sequential(
            nn.Linear(30, 30),
            nn.ReLU()
            )
        self.fullyConnect3 = nn.Sequential(
            nn.Linear(30, 3),
            nn.Softmax()
            )

    def forward(self, x):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = out.reshape(30, 3072)
        out = self.fullyConnect1(out)
        out = self.fullyConnect2(out)
        out = self.fullyConnect3(out)
        return out

H = torch.rand(
test = rnaConvNet()
out = test(H)
