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


def to_tensor(x):
    return torch.tensor(x).float()


# +
def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


# -

class Diffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        self.register_buffer("betas", to_tensor(betas))
        self.register_buffer("std_fwd", to_tensor(std_fwd))
        self.register_buffer("std_bwd", to_tensor(std_bwd))
        self.register_buffer("mu_x0", to_tensor(mu_x0))
        self.register_buffer("mu_x1", to_tensor(mu_x1))
        self.register_buffer("var", to_tensor(var))
        self.register_buffer("std_sb", to_tensor(std_sb))

        self.diffusion_steps = betas.shape[0]

    def q_sample(self, step, x_0, x_1):
        batch, *xdim = x_0.shape
        
        std_fwd = self.std_fwd[step].view(-1, 1)
        mu_x0 = self.mu_x0[step].view(-1, 1)
        mu_x1 = self.mu_x1[step].view(-1, 1)
        std = self.std_sb[step].view(-1, 1)
        x_t = mu_x0 * x_0 + mu_x1 * x_1 + std * torch.randn_like(x_0)
        label = (x_t - x_0) / std_fwd
        return x_t, label, std_fwd

    def sample_step(self, step, x, x0_fn, cond):
        std_n = self.std_fwd[step]
        std_pn = self.std_fwd[step-1]
        std_delta = (std_n**2 - std_pn**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_pn, std_delta)
        x_0 = x0_fn(x, step, cond)
        xt_prev = mu_x0 * x_0 + mu_xn * x
        if step > 1:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)        
        return xt_prev

    def sample(self, x, cond, x0_fn):
        for diff_step in reversed(range(1, self.betas.shape[0])):
            x = self.sample_step(diff_step, x, x0_fn, cond)
        return x
