# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from functools import partial
import torch
import torch.nn as nn

class Diffusion(nn.Module):
    def __init__(self, beta_min, beta_max):
        super().__init__()
        self.beta_max = beta_max
        self.beta_min = beta_min

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def drift(self, x, t):
        beta_t = self.beta(t)
        return -0.5 * beta_t * x

    def diffusion(self, x, t):
        return torch.sqrt(self.beta(t))
    
    def marginal(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def q_sample(self, t, x):
        batch, *xdim = x.shape

        noise = torch.randn_like(x)
        mean, std = self.marginal(x, t)
        x_noisy = mean + std * noise
        return x_noisy, noise, std

    def sample_step(self, model, t, x, cond, dt):
        diff = self.diffusion(x, t)
        drift = self.drift(x, t) - diff * diff * model(x, t, cond)
        x = x - drift * dt
        x += diff * torch.randn_like(x) * np.sqrt(dt)        
        return x

    def sample(self, model, x, cond, diffusion_steps=150):
        dt = 1. / diffusion_steps
        for diff_step in reversed(range(0, diffusion_steps)):
            t = torch.tensor((diff_step + 1) / diffusion_steps).unsqueeze(-1).to(x)
            x = self.sample_step(model, t, x, cond, dt)
        return x