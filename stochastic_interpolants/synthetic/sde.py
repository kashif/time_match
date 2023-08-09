import numpy as np
import torch
import torch.nn as nn
from torchsde import sdeint
from architectures import Net

EPS = 1e-4

class SDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, epsilon):
        super().__init__()
        self.drift = drift
        self.epsilon = epsilon

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t[:, :1]) * t.clamp(0., 1.)
        drift = self.drift(x_t, t)
        return drift

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        return torch.ones_like(x_t) * np.sqrt(2 * self.epsilon)

class EpsilonSDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, gamma, epsilon):
        super().__init__()
        self.drift = drift
        self.gamma = gamma
        self.epsilon = epsilon

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t[:, :1]) * t.clamp(0., 1.)
        drift = self.drift(x_t, t)
        return drift

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        return torch.ones_like(x_t) * torch.sqrt(2 * self.epsilon * self.gamma(t))