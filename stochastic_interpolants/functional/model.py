import numpy as np
import torch
import torch.nn as nn
from torchsde import sdeint
from utils import get_gp_covariance
from architectures import *

EPS = 1e-4
class SDE(nn.Module):
    noise_type = 'additive'
    sde_type = 'ito'

    def __init__(self, drift, score, ts, epsilon=0.1, gp=False):
        super().__init__()
        self.epsilon = epsilon
        self.drift = drift
        self.score = score
        self.ts = ts
        self.gp = gp

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t).unsqueeze(-1) * t.clamp(0., 1.)
        x = torch.cat([x_t.unsqueeze(-1), self.ts], dim=-1)
        b_t = self.drift(x, t)
        s_t = self.score(x, t)
        drift = b_t + self.epsilon * s_t
        return drift.squeeze(-1)

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        bsz, state_size = x_t.shape
        if self.gp:
            L = get_gp_covariance(self.ts)
            L = torch.linalg.cholesky(L)
        else:
            L = torch.eye(self.ts.shape[1], self.ts.shape[1]).to(x_t).unsqueeze(0).repeat(bsz, 1, 1)
        L *= np.sqrt(2 * self.epsilon)
        return L

class JointSDE(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, model, ts, epsilon=0.1, gp=False):
        super().__init__()
        self.epsilon = epsilon
        self.model = model
        self.ts = ts
        self.gp = gp

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t).unsqueeze(-1) * t.clamp(0., 1.)
        x = torch.cat([x_t.unsqueeze(-1), self.ts], dim=-1)
        out = self.model(x, t)
        b_t = out[:, :, :1]
        s_t = out[:, :, 1:]
        drift = b_t + self.epsilon * s_t
        return drift.squeeze(-1)

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        bsz, state_size = x_t.shape
        if self.gp:
            L = get_gp_covariance(self.ts)
            L = torch.linalg.cholesky(L)
        else:
            L = torch.eye(self.ts.shape[1], self.ts.shape[1]).to(x_t).unsqueeze(0).repeat(bsz, 1, 1)
        L *= np.sqrt(2 * self.epsilon)
        return L

class ODE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, model, ts):
        super().__init__()
        self.model = model
        self.ts = ts

    # Drift
    @torch.no_grad()
    def f(self, t, x_t):
        t = torch.ones_like(x_t).unsqueeze(-1) * t.clamp(0., 1.)
        x = torch.cat([x_t.unsqueeze(-1), self.ts], dim=-1)
        out = self.model(x, t)
        return out.squeeze(-1)

    # Diffusion
    @torch.no_grad()
    def g(self, t, x_t):
        return torch.zeros_like(x_t)

class SI_Indep(nn.Module):
    def __init__(self, interpolant, gp=False):
        super(SI_Indep, self).__init__()
        self.drift = BasicTransformerModel()
        self.score = BasicTransformerModel()
        self.interpolant = interpolant
        self.gp = gp

    def drift_fn(self, x, t, ts):
        x = torch.cat([x, ts], dim=-1)
        return self.drift(x, t)

    def score_fn(self, x, t, ts):
        x = torch.cat([x, ts], dim=-1)
        return self.score(x, t)

    def drift_loss(self, x_0, x_1, t, ts):
        with torch.no_grad():
            if self.gp:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t, ts=ts)
            else:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t)
            inter_ = inter - 2 * self.interpolant.gamma(t) * z

            dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)
        
        dt_inter = torch.cat([dt_inter, dt_inter], dim=0)
        z = torch.cat([z, -z], dim=0)
        inter = torch.cat([inter, inter_], dim=0)
        t = torch.cat([t, t], dim=0)
        ts = torch.cat([ts, ts], dim=0)

        field = self.drift_fn(inter, t, ts)
        # l = 0.5 * (field ** 2).sum(dim=-1) - ((dt_inter + self.interpolant.gamma_(t) * z) * field).sum(dim=-1)
        l = 0.5 * ((field - (dt_inter + self.interpolant.gamma_(t) * z))**2).sum(dim=-1)
        return l

    def score_loss(self, x_0, x_1, t, ts):
        with torch.no_grad():
            if self.gp:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t, ts=ts)
            else:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t)
            inter_ = inter - 2 * self.interpolant.gamma(t) * z

        z = torch.cat([z, -z], dim=0)
        inter = torch.cat([inter, inter_], dim=0)
        t = torch.cat([t, t], dim=0)
        ts = torch.cat([ts, ts], dim=0)

        score = self.score_fn(inter, t, ts)
        # l = 0.5 * (score ** 2).sum(dim=-1) + ((z / (self.interpolant.gamma(t) + EPS)) * score).sum(dim=-1)
        l = 0.5 * ((score + (z / (self.interpolant.gamma(t) + EPS))) ** 2).sum(dim=-1)
        return l
    
    def train_step(self, x_0, x_1, t, ts):
        loss = self.drift_loss(x_0, x_1, t, ts) + self.score_loss(x_0, x_1, t, ts)
        return loss
        
    @torch.no_grad()
    def get_sde(self, epsilon, ts, T=None):
        return SDE(self.drift, self.score, ts, epsilon, gp=self.gp)

    @torch.no_grad()
    def sample(self, x, ts, epsilon=0.01):
        bsz, T, dim = x.shape
        sde = self.get_sde(epsilon, ts, T=T)
        t = torch.linspace(0., 1., 5).to(x)
        samples = sdeint(sde, x.view(bsz, T), t, dt=1e-3)[-1]
        return samples

    def forward(self, x_0, x_1, t, ts):
        return self.train_step(x_0, x_1, t, ts)

class SI_Dep(nn.Module):
    def __init__(self, interpolant, gp=False):
        super(SI_Dep, self).__init__()
        self.model = BasicTransformerModel(out_dim=2)
        self.interpolant = interpolant
        self.gp = gp

    def model_fn(self, x, t, ts):
        x = torch.cat([x, ts], dim=-1)
        return self.model(x, t)

    def loss(self, x_0, x_1, t, ts):
        with torch.no_grad():
            if self.gp:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t, ts=ts)
            else:
                inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t)
            inter_ = inter - 2 * self.interpolant.gamma(t) * z
            dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)

        dt_inter = torch.cat([dt_inter, dt_inter], dim=0)
        z = torch.cat([z, -z], dim=0)
        inter = torch.cat([inter, inter_], dim=0)
        t = torch.cat([t, t], dim=0)
        ts = torch.cat([ts, ts], dim=0)

        out = self.model_fn(inter, t, ts)
        l = 0.5 * ((out[:, :, :1] - (dt_inter + self.interpolant.gamma_(t) * z))**2).sum(dim=-1)
        l += 0.5 * ((out[:, :, 1:] + (z / (self.interpolant.gamma(t) + EPS)))**2).sum(dim=-1)
        # l += 0.5 * ((self.interpolant.gamma(t) * out[:, :, 1:] + z)**2).sum(dim=-1)
        return l
    
    def train_step(self, x_0, x_1, t, ts):
        loss = self.loss(x_0, x_1, t, ts)
        return loss
        
    @torch.no_grad()
    def get_sde(self, epsilon, ts, T=None):
        return JointSDE(self.model, ts, epsilon, gp=self.gp)

    @torch.no_grad()
    def sample(self, x, ts, epsilon=0.01):
        bsz, T, dim = x.shape
        sde = self.get_sde(epsilon, ts, T=T)
        t = torch.linspace(0., 1., 5).to(x)
        samples = sdeint(sde, x.view(bsz, T), t, dt=1e-3)[-1]
        return samples

    def forward(self, x_0, x_1, t, ts):
        return self.train_step(x_0, x_1, t, ts)

class FlowMatching(nn.Module):
    def __init__(self):
        super(FlowMatching, self).__init__()
        self.model = BasicTransformerModel(out_dim=1)

    def model_fn(self, x, t, ts):
        x = torch.cat([x, ts], dim=-1)
        return self.model(x, t)

    def loss(self, x_0, x_1, t, ts):
        x_t = t * x_1 + (1-t) * x_0
        out = self.model_fn(x_t, t, ts)
        l = ((out - (x_1 - x_0)) ** 2).sum(dim=-1)
        return l
    
    def train_step(self, x_0, x_1, t, ts):
        loss = self.loss(x_0, x_1, t, ts)
        return loss
        
    @torch.no_grad()
    def get_sde(self, ts, T=None):
        return ODE(self.model, ts)

    @torch.no_grad()
    def sample(self, x, ts):
        bsz, T, dim = x.shape
        sde = self.get_sde(ts, T=T)
        t = torch.linspace(0., 1., 5).to(x)
        samples = sdeint(sde, x.view(bsz, T), t, dt=1e-3)[-1]
        return samples

    def forward(self, x_0, x_1, t, ts):
        return self.train_step(x_0, x_1, t, ts)

class DDPM(nn.Module):
    def __init__(self, alphas, betas, diffusion_steps, gp=False):
        super(DDPM, self).__init__()
        self.score = BasicTransformerModel()
        self.diffusion_steps = diffusion_steps
        self.alphas = alphas
        self.betas = betas
        self.gp = gp
    
    def add_noise(self, x, t, ts=None):
        """
        x: Clean data sample, shape [B, S, D]
        t: Diffusion step, shape [B, S, 1]
        ts: Times of observations, shape [B, S, 1]
        """
        noise = torch.randn_like(x)
        
        if self.gp:
            cov = get_gp_covariance(ts)
            L = torch.linalg.cholesky(cov)
            noise = L @ noise
        
        alpha = self.alphas[t.long()].to(x)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        
        return x_noisy, noise

    def denoise_fn(self, x, t, ts):
        x = torch.cat([x, ts], dim=-1)
        return self.score(x, t / (self.diffusion_steps - 1))

    def train_step(self, x, ts, t):
        x_noisy, noise = self.add_noise(x, t, ts)
        pred_noise = self.denoise_fn(x_noisy, t, ts)
        
        loss = (pred_noise - noise)**2
        return torch.mean(loss)

    @torch.no_grad()
    def sample(self, ts):
        x = torch.randn_like(ts)
        if self.gp:
            cov = get_gp_covariance(ts)
            L = torch.linalg.cholesky(cov)
            x = L @ x
        
        for diff_step in reversed(range(0, self.diffusion_steps)):
            alpha = self.alphas[diff_step]
            beta = self.betas[diff_step]

            z = torch.randn_like(ts)
            if self.gp:
                z = L @ z
            
            pred_noise = self.denoise_fn(x, diff_step, ts)
            
            x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() 
            if diff_step > 0:
                x += beta.sqrt() * z
        return x
