import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from interpolants import *
from coupling import *
from gamma import *
import ot as pot
import seaborn as sns

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

def plot(sol, figsize=(24, 4)):
    sol = sol.detach().cpu()
    fig, axs = plt.subplots(ncols=sol.shape[0], figsize=figsize)
    f, ax = plt.subplots(figsize=(10, 10))
    
    for i in range(sol.shape[0]):
        axs[i].scatter(sol[i, :, 0], sol[i, :, 1], s=2)
        axs[i].axis('off')
    ax.scatter(sol[-1, :, 0], sol[-1, :, 1], s=2)
    ax.axis('off')
    return fig, f

def kde_plot(sol, figsize=(10, 10)):
    sol = sol.detach().cpu()
    fig, ax = plt.subplots(figsize=figsize)
    kplot = sns.kdeplot(x=sol[:,0], y=sol[:,1], fill=True,
        thresh=0, levels=100, cmap='crest_r', ax=ax, cut=12)
    ax.set_xlim(-6., 6.)
    ax.axis('off')
    return fig

def displacement_plot(sol, figsize=(10, 10)):
    sol = sol[:, :1000, :]
    sol = sol.detach().cpu()
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(sol.shape[1]):
        ax.plot(sol[:, i, 0], sol[:, i, 1], c='olive', alpha=0.1)
    ax.scatter(sol[0, :, 0], sol[0, :, 1], s=5, c='black', alpha=1.,linewidths=0.5)
    ax.scatter(sol[-1, :, 0], sol[-1, :, 1], s=5, c='blue', alpha=1., linewidths=0.5)
    ax.axis('off')
    return fig

def get_data(source, target, batch_size, device):
    idx_1 = np.random.choice(source.shape[0], size=batch_size)
    idx_2 = np.random.choice(target.shape[0], size=batch_size)
    x_0 = source[idx_1].to(device)
    x_1 = target[idx_2].to(device)
    t = torch.rand(batch_size).unsqueeze(-1).to(x_0)
    return x_0, x_1, t

def get_ot_sample(x_0, x_1):
    a, b = pot.unif(x_0.shape[0]), pot.unif(x_1.shape[0])
    M = torch.cdist(x_0, x_1) ** 2
    M = M / M.max()
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=x_0.shape[0])
    i, j = np.divmod(choices, pi.shape[1])
    return x_0[i], x_1[j]

if __name__ == '__main__':
    from data import *
    source = torch.tensor(inf_train_gen('8gaussians', batch_size=256)).float().detach().cpu()
    target = torch.tensor(inf_train_gen('moons', batch_size=256)).float().detach().cpu()

    x_0, x_1 = get_ot_sample(source, target)
    for i in range(x_0.shape[0]):
        plt.plot([x_0[i ,0], x_1[i, 0]], [x_0[i, 1], x_1[i, 1]], c='olive', alpha=0.1)
    plt.scatter(x_0[:, 0], x_0[:, 1], s=2)
    plt.scatter(x_1[:, 0], x_1[:, 1], s=2)
    plt.savefig('trial.png')