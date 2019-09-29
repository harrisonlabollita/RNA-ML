# CONVOLUTIONAL NEURAL NETWORK IN PYTORCH
# Author: Harrison LaBollita
# Version: 2.0

import numpy as np
import torch
import torchvision
import torch.nn.functional as F



class rnaConvNet(torch.nn.Module):

    def __init__(self, seq_length, num_classes, batch_size):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_classes = num_classes

        super(rnaConvNet, self).__init__()

        self.convLayer1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        self.convLayer2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        self.fullyConnect1 = torch.nn.Linear( 16*self.seq_length, self.seq_length)

        self.fullyConnect2 = torch.nn.Linear(self.seq_length, self.seq_length)

        self.fullyConnect3 = torch.nn.Linear(self.seq_length, self.num_classes)




    def forward(self, x):
        # Output size = (batch_size, 16, seq_length +1 , seq_length + 1 )
        out = F.relu(self.convLayer1(x))
        out = self.pool1(out)

        # Output size = (batch_size, 16, seq_length, seq_length)
        out = F.relu(self.convLayer2(out))
        out = self.pool2(out)

        # Output size = (batch_size, seq_length, seq_length*16)
        out = out.view(self.batch_size, self.seq_length, -1)

        # Output size = (batch_size, seq_length, seq_length)
        out = F.relu(self.fullyConnect1(out))

        # Output size = (batch_size, seq_length, seq_length)
        out = F.relu(self.fullyConnect2(out))

        # Output size = (batch_size, seq_length, num_classes)
        out = self.fullyConnect3(out)

        return out
