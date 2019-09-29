# Model for adjusting the output of RNAfold and fixing mistakes
import numpy as np
import torch
import torchvision
import torch.nn.functional as F






















def Loss():
    loss = torch.nn.BCEWithLogitsLoss()
    return loss

def Optimizer(net, learningRate):
    optimizer = optim.Adam(net.parameters(), learningRate)
    return optimizer
