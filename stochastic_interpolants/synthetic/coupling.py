import math
import numpy as np
import torch
import torch.nn as nn

EPS = 1e-4

class Coupling(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, out_dim=2):
        super(Coupling, self).__init__()
        in_dim=2
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self.reset_params()
    
    def reset_params(self):
        nn.init.zeros_(self.model[-1].weight)
    
    def forward(self, x_0, x_1):
        # x = torch.cat([x_0, x_1], dim=-1)
        x = 0.5 * (self.model(x_0) + self.model(x_1))
        return self.model(x)

class QuadCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(QuadCoupling, self).__init__()

        self.coupling = Coupling(2 * dim, hidden_dim, dim)
    
    def forward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return t * (1. - t) * x

    def backward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return (1 - 2. * t) * x

class SqrtCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(SqrtCoupling, self).__init__()

        self.coupling = Coupling(2 * dim, hidden_dim, dim)
    
    def forward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return torch.sqrt(2 * t * (1. - t)) * x

    def backward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return (1 - 2. * t) * x / (EPS + torch.sqrt(2 * t * (1. - t)))

class TrigCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=32):
        super(TrigCoupling, self).__init__()

        self.coupling = Coupling(2 * dim, hidden_dim, dim)
    
    def forward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return (torch.sin(math.pi * t) ** 2) * x

    def backward(self, x_0, x_1, t):
        x = self.coupling(x_0, x_1)
        return 2 * math.pi * torch.sin(math.pi * t) * torch.cos(math.pi * t) *  x

class FourierCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=32, M=5):
        super(FourierCoupling, self).__init__()

        self.dim = dim
        self.M = M
        self.coupling = Coupling(2 * dim, hidden_dim, dim * M)
        mpi = (torch.arange(1, M+1) * math.pi).view(1, M, 1)
        self.register_buffer('mpi', mpi)

    def forward(self, x_0, x_1, t):
        bsz = x_0.shape[0]
        t = t.unsqueeze(-1)
        x = self.coupling(x_0, x_1).view(bsz, self.M, self.dim) / self.M
        x = torch.sin(self.mpi * t) * x
        return x.sum(dim=1)

    def backward(self, x_0, x_1, t):
        bsz = x_0.shape[0]
        t = t.unsqueeze(-1)
        x = self.coupling(x_0, x_1).view(bsz, self.M, self.dim) / self.M
        x = self.mpi * torch.cos(self.mpi * t) * x
        return x.sum(dim=1)