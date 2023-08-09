import torch

@torch.no_grad()
def get_gp_covariance(t, gp_sigma = 0.1):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5 # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag