# Model for adjusting the output of RNAfold and fixing mistakes
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F


class foldNet(torch.nn.Module):

    def __init__(self, seq_length, hidden_size):
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        super(foldNet, self).__init__()

        self.fullyConnect1 = torch.nn.Linear(self.seq_length, self.hidden_size)
        self.fullyConnect2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fullyConnect3 = torch.nn.Linear(self.hidden_size, self.seq_length)


    def forward(self, x):

        out = F.relu(self.fullyConnect1(x))
        out = F.relu(self.fullyConnect2(out))
        out = self.fullyConnect3(out)

        return out


def Loss():
    loss = torch.nn.MSELoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)
    return optimizer
