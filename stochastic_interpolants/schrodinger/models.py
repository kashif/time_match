import numpy as np
import torch
import torch.nn as nn
from torchsde import sdeint
from architectures import Net
from sde import *

EPS = 1e-4

class Model(nn.Module):
    def __init__(self, interpolant, epsilon=1.):
        super(Model, self).__init__()

        self.model = Net()
        self.epsilon = epsilon
        self.interpolant = interpolant

    def train_step(self, x_0, x_1, t):
        inter, z = self.interpolant.interpolate(x_0, x_1, t)
        inter_ = inter - 2 * self.interpolant.gamma(t) * z

        dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.model(inter, t)
        field_ = self.model(inter_, t)

        l = 0.5 * (field ** 2 + field_ ** 2).sum(dim=-1)
        l -= (dt_inter * (field + field_)).sum(dim=-1)
        l += (self.interpolant.gamma_(t) * z * (field_ - field)).sum(dim=-1)
        l += (self.epsilon * z * (field - field_)).sum(dim=-1) / (self.interpolant.gamma(t).squeeze(-1) + EPS)
        return 0.5 * l
    
    @torch.no_grad()
    def sample(self, x_0, adaptive=True, dt=1e-3):
        sde = SDE(self.model, self.epsilon)
        t = torch.linspace(0., 1., 5).to(x_0)
        samples = sdeint(sde, x_0, t, adaptive=adaptive, dt=dt)
        return samples

class EpsilonModel(nn.Module):
    def __init__(self, interpolant, epsilon=1.):
        super(EpsilonModel, self).__init__()

        self.model = Net()
        self.epsilon = epsilon
        self.interpolant = interpolant

    def train_step(self, x_0, x_1, t):
        inter, z = self.interpolant.interpolate(x_0, x_1, t)
        inter_ = inter - 2 * self.interpolant.gamma(t) * z

        dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.model(inter, t)
        field_ = self.model(inter_, t)

        l = 0.5 * (field ** 2 + field_ ** 2).sum(dim=-1)
        l -= (dt_inter * (field + field_)).sum(dim=-1)
        l += (self.interpolant.gamma_(t) * z * (field_ - field)).sum(dim=-1)
        l += (self.epsilon * z * (field - field_)).sum(dim=-1)
        return 0.5 * l
    
    @torch.no_grad()
    def sample(self, x_0, adaptive=True, dt=1e-3):
        sde = EpsilonSDE(self.model, self.interpolant.gamma, self.epsilon)
        t = torch.linspace(0., 1., 5).to(x_0)
        samples = sdeint(sde, x_0, t, adaptive=adaptive, dt=dt)
        return samples