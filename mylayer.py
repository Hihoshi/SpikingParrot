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


class Surrogate_Dyt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma):
        ctx.save_for_backward(x, alpha, gamma)
        return (x > 0).float()  # 前向输出阶跃函数

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, gamma = ctx.saved_tensors
        tanh_ax = torch.tanh(alpha * x)
        derivative = gamma * alpha * (1 - tanh_ax**2)  # 替代梯度
        grad_x = grad_output * derivative
        
        # 计算alpha和gamma的梯度，按输入维度聚合
        sum_dims = list(range(grad_output.dim() - 1))  # 聚合除最后一个维度外的所有维度
        grad_alpha = (grad_output * gamma * x * (1 - tanh_ax**2)).sum(dim=sum_dims)
        grad_gamma = (grad_output * alpha * (1 - tanh_ax**2)).sum(dim=sum_dims)
        
        return grad_x, grad_alpha, grad_gamma


class surrogate_dyt(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.alpha = nn.Parameter(torch.ones(input_size))
        self.gamma = nn.Parameter(torch.ones(input_size))
    
    def forward(self, x):
        # 确保alpha和gamma与x的最后一维对齐（例如input_size）
        return Surrogate_Dyt.apply(x, self.alpha, self.gamma)
