import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from interpolants import *
from coupling import *
from gamma import *

beta = torch.distributions.beta.Beta(5., 1.)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_gamma(gamma):
    if gamma == 'Trig':
        return TrigGamma()
    elif gamma == 'Quad':
        return QuadGamma()
    elif gamma == 'Sqrt':
        return SqrtGamma()
    else:
        return Gamma()

def get_interpolant(interpolant, gamma):
    if interpolant == 'Linear':
        return Linear(gamma)
    elif interpolant == 'Trig':
        return Trigonometric(gamma)
    elif interpolant == 'EncDec':
        return EncDec(gamma)

def get_coupling(coupling, dim, hidden_dim, M):
    if coupling == 'Quad':
        return QuadCoupling(dim, hidden_dim)
    elif coupling == 'Sqrt':
        return SqrtCoupling(dim, hidden_dim)
    elif coupling == 'Trig':
        return TrigCoupling(dim, hidden_dim)
    elif coupling == 'Fourier':
        return FourierCoupling(dim, hidden_dim, M)

def get_learned_interpolant(interpolant, coupling, gamma, dim=2, hidden_dim=32, M=5):
    interp = get_interpolant(interpolant, gamma)
    coupl = get_coupling(coupling, dim, hidden_dim, M)
    return LearnedInterpolant(interp, coupl)

def plot(sol, figsize=(20, 4)):
    sol = sol.detach().cpu()
    fig, axs = plt.subplots(ncols=sol.shape[0], figsize=figsize)
    
    for i in range(sol.shape[0]):
        axs[i].scatter(sol[i, :, 0], sol[i, :, 1], s=2)
    return fig

def get_data(source, target, batch_size, device):
    idx_1 = np.random.choice(target.shape[0], size=batch_size)
    x_1 = target[idx_1].to(device)
    x_0 = source.sample(batch_size).to(device)
    t = beta.sample((batch_size,)).unsqueeze(-1).to(x_0)
    # t = torch.rand(batch_size).unsqueeze(-1).to(x_0)
    return x_0, x_1, t