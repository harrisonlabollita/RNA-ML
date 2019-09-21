import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, dim, seq_len):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        A = torch.empty(self.dim, self.seq_len)
        torch.nn.init.xavier_normal_(A)
        self.A = torch.nn.Parameter(A)

    def forward(self, H):
        X = torch.nn.functional.softmax(torch.matmul(H, self.A), dim=-1)
        out = torch.matmul(X, H)
        return out


# test Attention
H = torch.rand(100, 30, 256)
atten = Attention(256, 30)
out = atten(H)
print(out)
