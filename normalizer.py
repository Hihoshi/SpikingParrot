import torch
import torch.nn as nn


class Dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size))
        self.gamma = nn.Parameter(torch.ones(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
    
    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta
