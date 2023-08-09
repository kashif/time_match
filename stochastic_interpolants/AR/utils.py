import torch
import numpy as np
import random
from interpolants import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

def i2sb_betas(n_timestep=1000, linear_start=1e-4, linear_end=1-2):
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

def get_interpolant(name):
    if name == 'trigODE':
        return TrigonometricODEInterpolants()
    elif name == 'EncDec':
        return EncDecInterpolant()