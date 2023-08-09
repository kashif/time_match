import torch
import torch.nn as nn

class AR(nn.Module):
    def __init__(self, T, std):
        self.std = std
        self.coeff = [0.8]
        self.T = T - 1
        
    @torch.no_grad()
    def ar(self, x):
        return (self.coeff[0] * x[:, -1] + self.std * torch.randn_like(x[:, -1])).unsqueeze(-1)

    @torch.no_grad()
    def get_gt_pred(self, y, n_samples, t=5):
        y_avg = y[:, :t]
        y_avg = torch.cat([y_avg, y[:, t-1:-1] * self.coeff[0]], dim=-1)
        return y_avg

    @torch.no_grad()
    def get_batch(self, batch_size):
        x = torch.randn(batch_size, 1)
        for t in range(self.T):
            x = torch.cat([x, self.ar(x)], dim=-1)
        return x