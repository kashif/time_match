from typing import List, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContDiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = diffusion_step * self.embedding
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim):
        dims = torch.arange(dim).unsqueeze(0).unsqueeze(0)  # [1,1,dim]
        return 10.0 ** (dims * 4.0 / dim)

class BasicTransformerModel(nn.Module):
    def __init__(self, dim=256, out_dim=1, heads=8, layers=4):
        super(BasicTransformerModel, self).__init__()
        self.time_embedding = nn.Sequential(
            ContDiffusionEmbedding(16, 32),
            nn.Linear(32, dim)
        )
        self.encoder = nn.Linear(2, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.model = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=layers),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x, t, clamp=True):
        x = self.encoder(x)
        if isinstance(t, float) or len(t.shape) < 2:
            t = torch.ones_like(x[...,:1]) * torch.tensor(t).to(x).clamp(0., 1.)
        t_emb = self.time_embedding(t)
        return self.model(x + t_emb)