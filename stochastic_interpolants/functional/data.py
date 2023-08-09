import torch
import numpy as np
from utils import get_gp_covariance

@torch.no_grad()
def get_sin(t):
    return torch.sin(10 * t + 2 * np.pi * torch.rand(t.shape[0], 1, 1).to(t))

@torch.no_grad()
def get_gp(t):
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)

    x = L @ torch.randn_like(t)
    return x

@torch.no_grad()
def get_linear(t):
    bsz, T, _ = t.shape
    w = torch.randn(bsz).view(bsz, 1, 1).to(t)
    b = torch.randn(bsz).view(bsz, 1, 1).to(t)
    return w * t + b

@torch.no_grad()
def get_quadratic(t):
    bsz, T, _ = t.shape
    a = torch.randn(bsz).view(bsz, 1, 1).to(t)
    b = torch.randn(bsz).view(bsz, 1, 1).to(t)
    c = torch.randn(bsz).view(bsz, 1, 1).to(t)
    return a * (t ** 2) + b * t + c

@torch.no_grad()
def get_data(t, dataset='sin'):
    if dataset == 'sin':
        return get_sin(t)
    elif dataset == 'gp':
        return get_gp(t)
    elif dataset == 'linear':
        return get_linear(t)
    elif dataset == 'quadratic':
        return get_quadratic(t)