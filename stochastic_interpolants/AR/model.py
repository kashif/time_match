import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from torchsde import sdeint
from utils import compute_gaussian_product_coef

EPS = 1e-6

class ODE(nn.Module):
    def __init__(self, drift, score):
        super().__init__()
        self.drift = drift
        self.score = score

    # Drift
    def f(self, t, x_t):
        t = torch.ones_like(x_t[:, :1]) * t.clamp(0., 1.)
        b_t = self.drift(x_t, t)
        b_t = torch.cat([b_t, torch.zeros_like(x_t[:, 1:])], dim=-1)
        return b_t
    
class SDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, drift, score, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.drift = drift
        self.score = score

    # Drift
    def f(self, t, x_t):
        t = torch.ones_like(x_t[:, :1]) * t.clamp(0., 1.)
        b_t = self.drift(x_t, t)
        s_t = self.score(x_t, t)
        drift = b_t + self.epsilon * s_t
        drift = torch.cat([drift, torch.zeros_like(x_t[:, 1:])], dim=-1)
        return drift

    # Diffusion
    def g(self, t, x_t):
        diff = torch.zeros_like(x_t)
        diff[:, :1] = np.sqrt(2 * self.epsilon)
        return diff

class SGMSDE(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, beta_min, beta_max, score, ode):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.score = score
        self.ode = ode

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    # Drift
    def f(self, t, x_t):
        t = 1. - t
        drift = 0.5 * self.beta(t) * x_t[:, :1] + self.beta(t) * self.score(x_t, t)
        drift = torch.cat([drift, torch.zeros_like(x_t[:, 1:])], dim=-1)
        return drift

    # Diffusion
    def g(self, t, x_t):
        t = 1. - t
        diff = torch.zeros_like(x_t)
        if not self.ode:
            diff[:, :1] = torch.sqrt(self.beta(t))
        return diff

class Net(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x, t):
        if isinstance(t, int) or isinstance(t, float) or len(t.shape) < 2:
            t = torch.ones_like(x[..., 0].unsqueeze(-1) * t)
        t = t.to(x)
        x = torch.cat([x, t], dim=-1)
        return self.model(x)

class DDPM(nn.Module):
    def __init__(self, alphas, betas, diffusion_steps, in_dim=3):
        super(DDPM, self).__init__()
        
        self.alphas = alphas
        self.betas = betas
        self.diffusion_steps = diffusion_steps
        self.model = Net(in_dim)

    def train_step(self, x, cond, t):
        noise = torch.randn_like(x)
        alpha = self.alphas[t.long()].to(x)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
        pred_noise = self.model(torch.cat([x_noisy, cond], dim=-1), t / (self.diffusion_steps-1))
    
        loss = (pred_noise - noise)**2
        return torch.mean(loss)
    
    @torch.no_grad()
    def sample_step(self, x, cond, step, ode=False):
        alpha = self.alphas[step]
        beta = self.betas[step]

        z = torch.randn_like(x)
        t = torch.ones_like(x) * step / (self.diffusion_steps - 1)
        pred_noise = self.model(torch.cat([x, cond], dim=-1), t)
        
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt()
        if step > 0 and not ode:
            x += beta.sqrt() * z
        
        return x
    
    @torch.no_grad()
    def sample(self, cond, ode=False):
        x = torch.randn_like(cond[:, :1])
        for diff_step in reversed(range(0, self.diffusion_steps)):
            x = self.sample_step(x, cond, diff_step, ode=ode)
        return x

    @torch.no_grad()
    def forecast(self, x, T, n_samples=50, ode=False):
        bsz = x.shape[0]
        if ode:
            n_samples = 1

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = x[:, ts-1].unsqueeze(-1)
            pred = torch.cat([pred, self.sample(cond, ode=ode)], dim=-1)
        return pred.view(n_samples, bsz, -1)

class I2SB(nn.Module):
    def __init__(self, betas, std_fwd, std_bwd, std_sb, mu_x0, mu_x1, in_dim=3):
        super(I2SB, self).__init__()
        
        self.betas = betas
        self.std_fwd = std_fwd
        self.std_bwd = std_bwd
        self.std_sb  = std_sb
        self.mu_x0 = mu_x0
        self.mu_x1 = mu_x1

        self.model = Net(in_dim)

    def train_step(self, x_1, x_0, cond, t, ode=False):
        std_fwd = self.std_fwd[t]

        x_t = self.mu_x0[t] * x_0 + self.mu_x1[t] * x_1
        if not ode:
            x_t = x_t + self.std_sb[t] * torch.randn_like(x_t)
        label = (x_t - x_0) / std_fwd
        
        pred = self.model(x_t, t)
        loss = (pred - label) ** 2
        return torch.mean(loss)

    @torch.no_grad()
    def compute_pred_x0(self, x_t, t):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.std_fwd[t]
        pred_x0 = x_t - std_fwd * self.model(x_t, t)
        return pred_x0

    @torch.no_grad()
    def sample_step(self, x_t, step, ode=False):
        std_n = self.std_fwd[step]
        std_pn = self.std_fwd[step-1]
        std_delta = (std_n**2 - std_pn**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_pn, std_delta)
        x_0 = self.compute_pred_x0(x_t, step)
        xt_prev = mu_x0 * x_0 + mu_xn * x_t
        if not ode and step > 1:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)        
        return xt_prev
    
    @torch.no_grad()
    def sample(self, x_T, ode=False):
        x = x_T
        for diff_step in reversed(range(1, self.betas.shape[0])):
            x = self.sample_step(x, diff_step, ode=ode)
        return x

    @torch.no_grad()
    def forecast(self, x, T, n_samples=50, ode=False):
        bsz = x.shape[0]
        if ode:
            n_samples = 1

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = x[:, ts-1].unsqueeze(-1)
            pred = torch.cat([pred, self.sample(cond, ode=ode)], dim=-1)
        return pred.view(n_samples, bsz, -1)

class ConditionalStochasticInterpolant(nn.Module):
    def __init__(self, interpolant, in_dim):
        super(ConditionalStochasticInterpolant, self).__init__()
        
        self.drift = Net(in_dim)
        self.score = Net(in_dim)
        self.interpolant = interpolant        

    def drift_loss(self, x_0, x_1, t, cond=None):
        inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t)
        inter_ = inter - 2 * self.interpolant.gamma(t) * z

        dt_inter = self.interpolant.dt_interpolate(x_0, x_1, t)

        if cond is not None:
            inter = torch.cat([inter, cond], dim=-1)
            inter_ = torch.cat([inter_, cond], dim=-1)

        field = self.drift(inter, t)
        field_ = self.drift(inter_, t)

        l = 0.5 * (field ** 2).sum(dim=-1) - ((dt_inter + self.interpolant.gamma_(t) * z) * field).sum(dim=-1)
        l += 0.5 * (field_ ** 2).sum(dim=-1) - ((dt_inter - self.interpolant.gamma_(t) * z) * field_).sum(dim=-1)

        return 0.5 * l

    def score_loss(self, x_0, x_1, t, cond=None):
        inter, z = self.interpolant.noisy_interpolate(x_0, x_1, t)
        inter_ = inter - 2 * self.interpolant.gamma(t) * z

        if cond is not None:
            inter = torch.cat([inter, cond], dim=-1)
            inter_ = torch.cat([inter_, cond], dim=-1)

        score = self.score(inter, t)
        score_ = self.score(inter_, t)
        
        l = 0.5 * (score ** 2).sum(dim=-1) + ((z / (self.interpolant.gamma(t) + EPS)) * score).sum(dim=-1)
        l += 0.5 * (score_ ** 2).sum(dim=-1) - ((z / (self.interpolant.gamma(t) + EPS)) * score_).sum(dim=-1)
        
        return 0.5 * l

    def train_step(self, x_0, x_1, t, cond=None):
        loss = self.drift_loss(x_0, x_1, t, cond).mean() + self.score_loss(x_0, x_1, t, cond).mean()
        return loss

    @torch.no_grad()
    def sample(self, x, de, cond=None, ode=False, dt=1e-3):
        t = torch.linspace(0., 1., 2).to(x)
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)

        if ode:
            return odeint(de.f, x, t)[-1, :, :1]
        else:
            return sdeint(de, x, t, adaptive=True, dt=dt)[-1, :, :1]
    
    @torch.no_grad()
    def get_ode(self):
        return ODE(self.drift, self.score)

    @torch.no_grad()
    def get_sde(self, epsilon):
        return SDE(self.drift, self.score, epsilon)

    @torch.no_grad()
    def forecast_conditional(self, x, T, epsilon=0.25, n_samples=50, ode=False, dt=1e-3):
        bsz = x.shape[0]
        if ode:
            n_samples = 1
            de = self.get_ode()
        else:
            de = self.get_sde(epsilon)

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = x[:, ts-1].unsqueeze(-1)
            x_0 = torch.randn_like(cond)
            pred = torch.cat([pred, self.sample(x_0, de, cond=cond, ode=ode, dt=dt)], dim=-1)
        return pred.view(n_samples, bsz, -1)

    @torch.no_grad()
    def forecast_unconditional(self, x, T, epsilon=0.25, n_samples=50, ode=False, dt=1e-3):
        bsz = x.shape[0]
        if ode:
            n_samples = 1
            de = self.get_ode()
        else:
            de = self.get_sde(epsilon)

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = None
            x_0 = x[:, ts-1].unsqueeze(-1)
            pred = torch.cat([pred, self.sample(x_0, de, cond=cond, ode=ode, dt=dt)], dim=-1)
        return pred.view(n_samples, bsz, -1)

    @torch.no_grad()
    def forecast_conditional_new(self, x, T, epsilon=0.25, n_samples=50, ode=False, dt=1e-3):
        bsz = x.shape[0]
        if ode:
            n_samples = 1
            de = self.get_ode()
        else:
            de = self.get_sde(epsilon)

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = x[:, ts-1].unsqueeze(-1)
            x_0 = x[:, ts-1].unsqueeze(-1)
            pred = torch.cat([pred, self.sample(x_0, de, cond=cond, ode=ode, dt=dt)], dim=-1)
        return pred.view(n_samples, bsz, -1)

class SGM(nn.Module):
    def __init__(self, beta_min, beta_max, in_dim=3):
        super(SGM, self).__init__()
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.model = Net(in_dim)
    
    def compute_marginal(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def train_step(self, x, cond, t):
        mean, std = self.compute_marginal(x, t)
        noise = torch.randn_like(x)
        x_noisy = mean + std * noise
    
        score = self.model(torch.cat([x_noisy, cond], dim=-1), t)    
        loss = ((score + noise / std)**2).sum(dim=-1)
        return torch.mean(loss)
        
    @torch.no_grad()
    def sample(self, cond, ode=False):
        x = torch.randn_like(cond[:, :1])
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        t = torch.linspace(0., 1. - 1e-3, 2).to(x)
        sde = SGMSDE(self.beta_min, self.beta_max, self.model, ode=ode)
        return sdeint(sde, x, t, adaptive=True)[-1, :, :1]

    @torch.no_grad()
    def forecast(self, x, T, n_samples=50, ode=False):
        bsz = x.shape[0]
        if ode:
            n_samples = 1

        x = x.unsqueeze(0).repeat(n_samples, 1, 1).view(bsz * n_samples, -1)
        pred = x[:, :5]

        for ts in range(5, T):
            cond = x[:, ts-1].unsqueeze(-1)
            pred = torch.cat([pred, self.sample(cond, ode=ode)], dim=-1)
        return pred.view(n_samples, bsz, -1)
