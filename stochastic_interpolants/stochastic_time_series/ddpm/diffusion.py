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
    def __init__(self, betas):
        super().__init__()
        alphas = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("betas", torch.tensor(betas))
        self.register_buffer("alphas", torch.tensor(alphas))
        self.diffusion_steps = betas.shape[0]

    # def q_sample(self, step, x):
    #     B, T = x.shape[:2]
    #     noise = torch.randn_like(x)
    #     noisy_output = self.scheduler.add_noise(
    #         x.view(B * T, 1, -1), noise.view(B * T, 1, -1), step
    #     )
    #     return noisy_output, noise

    def q_sample(self, step, x):
        batch, *xdim = x.shape

        noise = torch.randn_like(x)
        alpha = self.alphas.to(x)[step.long()].unsqueeze(-1)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        return x_noisy, noise

    def sample_step(self, model, step, x, cond):
        alpha = self.alphas[step]
        beta = self.betas[step]
        bsz, _, _ = x.shape

        z = torch.randn_like(x)
        t = torch.ones(bsz).to(x) * step
        pred_noise = model(x, t, cond)
        
        x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt()
        if step > 0:
            x += beta.sqrt() * z
        
        return x

    def sample(self, model, x, cond):
        for diff_step in reversed(range(0, self.diffusion_steps)):
            x = self.sample_step(model, diff_step, x, cond)
        return x

    # def sample(self, model, sample, context):
    #     # context [B, T, H]
    #     B, T = context.shape[:2]
    #     print(sample.shape)
    #     print(context.shape)

    #     self.scheduler.set_timesteps(149)
    #     for t in self.scheduler.timesteps:
    #         model_output = model(sample, torch.ones(B*T).to(sample) * t, context.view(B * T, 1, -1))
    #         sample = self.scheduler.step(model_output, t, sample).prev_sample

    #     return sample.view(B, T, -1)