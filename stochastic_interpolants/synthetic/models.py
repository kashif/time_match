import numpy as np
import torch
import torch.nn as nn
from torchsde import sdeint
from architectures import Net
from sde import *

EPS = 1e-4

def laplace_like(x):
    return torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0])).sample(x.shape).squeeze(-1).to(x)

class SIModel(nn.Module):
    def __init__(self, interpolant, conditional_path='gaussian'):
        super(SIModel, self).__init__()

        self.drift = Net()
        self.score = Net()
        self.conditional_path = conditional_path
        self.interpolant = interpolant
    
    def noisy_interpolate(self, x_0, x_1, t):
        inter = self.interpolant.interpolate(x_0, x_1, t)
        if self.conditional_path == 'gaussian':
            z = torch.randn_like(inter)
        else:
            z = laplace_like(inter)

        return inter + self.interpolant.gamma(t) * z
    
    def train_drift_gaussian(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.drift(inter, t)
        field_ = self.drift(inter_, t)

        l = 0.5 * ((field - dt_interpol - self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)
        l += 0.5 * ((field_ - dt_interpol + self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)

        return l

    def train_drift_laplace(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = laplace_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.drift(inter, t)
        field_ = self.drift(inter_, t)

        l = 0.5 * ((field - dt_interpol - self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)
        l += 0.5 * ((field_ - dt_interpol + self.interpolant.gamma_(t) * z) ** 2).sum(dim=-1)

        return l

    def train_score_gaussian(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        score = self.score(inter, t)
        score_ = self.score(inter_, t)

        l = 0.5 * ((score + z / (self.interpolant.gamma(t) + EPS)) ** 2).sum(dim=-1)
        l += 0.5 * ((score_ - z / (self.interpolant.gamma(t) + EPS)) ** 2).sum(dim=-1)

        return l

    def train_score_laplace(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        score = self.score(inter, t)
        score_ = self.score(inter_, t)

        l = 0.5 * ((score + torch.sign(z) / (self.interpolant.gamma(t) + EPS)) ** 2).sum(dim=-1)
        l += 0.5 * ((score_ - torch.sign(z) / (self.interpolant.gamma(t) + EPS)) ** 2).sum(dim=-1)

        return l

    def train_step(self, x_0, x_1, t):
        if self.conditional_path == 'gaussian':
            loss = self.train_score_gaussian(x_0, x_1, t) + self.train_drift_gaussian(x_0, x_1, t)
        else:
            loss = self.train_score_laplace(x_0, x_1, t) + self.train_drift_laplace(x_0, x_1, t)
        return loss
    
    @torch.no_grad()
    def sample(self, x_0, adaptive=True, dt=1e-3, epsilon=1.):
        drift = lambda x, t: self.drift(x, t) + epsilon * self.score(x, t)
        sde = SDE(drift, epsilon)
        t = torch.linspace(0., 1., 5).to(x_0)
        samples = sdeint(sde, x_0, t, adaptive=adaptive, dt=dt)
        return samples

class SBModel(nn.Module):
    def __init__(self, interpolant, conditional_path='gaussian', epsilon=None):
        super(SBModel, self).__init__()

        self.model = Net()
        self.conditional_path = conditional_path
        self.interpolant = interpolant
        self.epsilon = epsilon
    
    def noisy_interpolate(self, x_0, x_1, t):
        inter = self.interpolant.interpolate(x_0, x_1, t)
        if self.conditional_path == 'gaussian':
            z = torch.randn_like(inter)
        else:
            z = laplace_like(inter)

        return inter + self.interpolant.gamma(t) * z
    
    def train_gaussian(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = torch.randn_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.model(inter, t)
        field_ = self.model(inter_, t)

        l = 0.5 * (field ** 2 + field_ ** 2).sum(dim=-1)
        l -= (dt_interpol * (field + field_)).sum(dim=-1)
        l += (self.interpolant.gamma_(t) * z * (field_ - field)).sum(dim=-1)
        l += (self.epsilon * z * (field - field_)).sum(dim=-1) / (self.interpolant.gamma(t).squeeze(-1) + EPS)
        return 0.5 * l

    def train_laplace(self, x_0, x_1, t):
        interpol = self.interpolant.interpolate(x_0, x_1, t)
        z = laplace_like(interpol)

        inter = interpol + self.interpolant.gamma(t) * z
        inter_ = interpol - self.interpolant.gamma(t) * z

        dt_interpol = self.interpolant.dt_interpolate(x_0, x_1, t)

        field = self.model(inter, t)
        field_ = self.model(inter_, t)

        l = 0.5 * (field ** 2 + field_ ** 2).sum(dim=-1)
        l -= (dt_interpol * (field + field_)).sum(dim=-1)
        l += (self.interpolant.gamma_(t) * z * (field_ - field)).sum(dim=-1)
        l += (self.epsilon * torch.sign(z) * (field - field_)).sum(dim=-1) / (self.interpolant.gamma(t).squeeze(-1) + EPS)
        return 0.5 * l

    def train_step(self, x_0, x_1, t):
        if self.conditional_path == 'gaussian':
            loss = self.train_gaussian(x_0, x_1, t)
        else:
            loss = self.train_laplace(x_0, x_1, t)
        return loss
    
    @torch.no_grad()
    def sample(self, x_0, adaptive=True, dt=1e-3, epsilon=1.):
        sde = SDE(self.model, epsilon)
        t = torch.linspace(0., 1., 5).to(x_0)
        samples = sdeint(sde, x_0, t, adaptive=adaptive, dt=dt)
        return samples

# class EpsilonModel(nn.Module):
#     def __init__(self, interpolant, epsilon=1.):
#         super(EpsilonModel, self).__init__()

#         self.model = Net()
#         self.epsilon = epsilon
#         self.interpolant = interpolant

#     def train_step(self, x_0, x_1, t):
#         inter, z = self.interpolant.interpolate(x_0, x_1, t)
#         inter_ = inter - 2 * self.interpolant.gamma(t) * z

#         dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)

#         field = self.model(inter, t)
#         field_ = self.model(inter_, t)

#         l = 0.5 * (field ** 2 + field_ ** 2).sum(dim=-1)
#         l -= (dt_inter * (field + field_)).sum(dim=-1)
#         l += (self.interpolant.gamma_(t) * z * (field_ - field)).sum(dim=-1)
#         l += (self.epsilon * z * (field - field_)).sum(dim=-1)
#         return 0.5 * l
    
#     @torch.no_grad()
#     def sample(self, x_0, adaptive=True, dt=1e-3):
#         sde = EpsilonSDE(self.model, self.interpolant.gamma, self.epsilon)
#         t = torch.linspace(0., 1., 5).to(x_0)
#         samples = sdeint(sde, x_0, t, adaptive=adaptive, dt=dt)
#         return samples