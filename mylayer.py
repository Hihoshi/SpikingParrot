import torch
import torch.nn as nn
import torch.nn.functional as F


class Dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size))
        self.gamma = nn.Parameter(torch.ones(input_size) * 2)
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.theta = nn.Parameter(torch.zeros(input_size))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x + self.beta) + self.theta


class Surrogate_Dyt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma):
        ctx.save_for_backward(x, alpha, gamma)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, gamma = ctx.saved_tensors
        tanh_ax = torch.tanh(alpha * x)
        derivative = gamma * alpha * (1 - tanh_ax**2)
        grad_x = grad_output * derivative
        
        sum_dims = list(range(grad_output.dim() - 1))
        grad_alpha = (grad_output * gamma * x * (1 - tanh_ax**2)).sum(dim=sum_dims)
        grad_gamma = (grad_output * alpha * (1 - tanh_ax**2)).sum(dim=sum_dims)
        
        return grad_x, grad_alpha, grad_gamma


class surrogate_dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size) * 2)
        self.gamma = nn.Parameter(torch.ones(input_size))
    
    def forward(self, x):
        # ensure aplha, gamma > 0
        alpha = F.softplus(self.alpha)
        gamma = F.softplus(self.gamma)
        return Surrogate_Dyt.apply(x, alpha, gamma)
