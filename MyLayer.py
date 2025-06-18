import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size) * math.log(math.expm1(1))) # equals to 1 after softplus
        self.gamma = nn.Parameter(torch.ones(input_size) * math.log(math.expm1(1)))
        self.beta = nn.Parameter(torch.zeros(input_size))
    
    def forward(self, x):
        # ensure positive alpha
        return F.softplus(self.gamma) * torch.tanh(F.softplus(self.alpha) * x) + self.beta


class Surrogate_Dyt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma):
        ctx.save_for_backward(x, alpha, gamma)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, gamma = ctx.saved_tensors
        alpha_plus = F.softplus(alpha)
        gamma_plus = F.softplus(gamma)

        tanh_ax = torch.tanh(alpha_plus * x)
        sech_2_ax = 1 - tanh_ax**2

        d_x = gamma_plus * alpha_plus * sech_2_ax
        d_alpha = gamma_plus * sech_2_ax * x * F.sigmoid(alpha)
        d_gamma = tanh_ax * F.sigmoid(gamma)
        
        grad_alpha = d_alpha * grad_output
        grad_gamma = d_gamma * grad_output
        grad_x = d_x * grad_output

        sum_dims = [d for d in range(grad_output.dim()) if d != grad_output.dim() - 1]
        if sum_dims:
            grad_alpha = grad_alpha.sum(dim=sum_dims)
            grad_gamma = grad_gamma.sum(dim=sum_dims)
        
        return grad_x, grad_alpha, grad_gamma


class surrogate_dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size) * math.log(math.expm1(1))) # equals to 1 after softplus
        self.gamma = nn.Parameter(torch.ones(input_size) * math.log(math.expm1(1)))
    
    def forward(self, x):
        return Surrogate_Dyt.apply(x, self.alpha, self.gamma)

