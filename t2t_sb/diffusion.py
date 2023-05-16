# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
from functools import partial
import torch


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class Diffusion:
    def __init__(self, betas):
        device = torch.device("cuda")
        betas_numpy = betas.cpu().numpy()

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas_numpy))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas_numpy))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """Sample q(x_t | x_0, x_1), i.e. eq 11"""

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(
        self, steps, pred_x0_fn, x1, cond, mask=None, ot_ode=False, log_steps=None
    ):
        xt = x1.detach()

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step, cond)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full(
                        (xt.shape[0],), prev_step, device=x1.device, dtype=torch.long
                    )
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1.0 - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach())
                xs.append(xt.detach())

        def stack_bwd_traj(z):
            return torch.flip(torch.stack(z, dim=1), dims=(1,))

        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
