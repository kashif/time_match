from typing import List, Callable
import numpy as np
import torch
import torch.nn as nn
import math

class BasicTransformerModel(nn.Module):
    def __init__(self, dim=256, out_dim=1, heads=8, layers=3):
        super(BasicTransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(3, dim),
            nn.TransformerEncoder(encoder_layer, num_layers=layers),
            nn.Linear(dim, out_dim)
        )
    
    def forward(self, x, t, clamp=True):
        if isinstance(t, float) or len(t.shape) < 2:
            t = torch.ones_like(x[...,:1]) * torch.tensor(t).to(x).clamp(0., 1.)
        x = torch.cat([x, t], dim=-1)
        return self.model(x)