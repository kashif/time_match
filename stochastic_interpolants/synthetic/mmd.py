import numpy as np
import os
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from data import inf_train_gen
from models import Model
from interpolants import *
from utils import *
from distributions import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='SI')
parser.add_argument('--gamma', type=str, default='Trig')
parser.add_argument('--interpolant', type=str, default='EncDec')
parser.add_argument('--coupling', type=str, default='Fourier')
parser.add_argument('--data', type=str, default='moons')
parser.add_argument('--epsilon', type=float, default=1.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(f'cuda:{args.device}')
print(args)

def mmd_rbf(X, Y, gamma=0.1):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)

if args.mode in ['SI', 'SI_Epsilon']:
    name = f'results/{args.mode}/{args.data}/{args.interpolant}_{args.gamma}/{args.seed}_{args.epsilon}'
    gamma = get_gamma(args.gamma)
    interpolant = get_interpolant(args.interpolant, gamma)
else:
    name = f'results/{args.mode}/{args.data}/{args.interpolant}_{args.gamma}/{args.coupling}/{args.seed}_{args.epsilon}'
    gamma = get_gamma(args.gamma)
    interpolant = get_learned_interpolant(args.interpolant, args.coupling, gamma, hidden_dim=32, M=5).to(device)

source = Gaussian(2).to(device)
model = Model(interpolant).to(device)
model.load_state_dict(torch.load(f'{name}/model.pt'))

true_data = torch.tensor(inf_train_gen(args.data, batch_size=1024)).float()
x_0 = source.sample(1024).to(device)
gen_data = model.sample(x_0)[-1].detach().cpu()
# gen_data = x_0.detach().cpu()
mmd = mmd_rbf(true_data, gen_data)
print(f'MMD: {mmd}')
# with open(f'{name}/mmd.txt', 'w') as f:
#     f.write(f'MMD: {mmd}')