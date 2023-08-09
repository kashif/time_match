import numpy as np
import os
import argparse
from time import time

import matplotlib.pyplot as plt

import torch
from data import inf_train_gen
from models import SIModel
from interpolants import *
from utils import *
from distributions import *
from metrics import wasserstein

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=str, default='Sqrt')
    parser.add_argument('--interpolant', type=str, default='Linear')
    parser.add_argument('--source', type=str, default='8gaussians')
    parser.add_argument('--target', type=str, default='moons')
    parser.add_argument('--noise', type=str, default='gaussian')
    parser.add_argument('--ot', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=100001)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    print(args)
    return args

def get_file_name(args):
    name = f'Plots/Toy'
    name = f'{name}/{args.target}_{args.noise}_{args.interpolant}_{args.gamma}'
    if args.ot:
        name = f'{name}_ot'
    return name

def get_folder_name(args):
    name = f'results/SI'
    if args.ot:
        name = f'{name}_ot'
    name = f'{name}/{args.source}_{args.target}/{args.interpolant}_{args.gamma}/{args.noise}/{args.seed}'
    return name

def get_interpolations(model, x_0, x_1, ts, device):
    interps = []
    for t in ts:
        interps.append(model.interpolant.interpolate(x_0, x_1, torch.ones_like(x_0[..., :1]) * t))
    interps = torch.stack(interps)
    fig, _ = plot(interps, (32, 4))
    return fig

def eval_step(model, x_0, x_1, adaptive=True, dt=1e-3, epsilon=1.):
    start = time()
    samples = model.sample(x_0, adaptive=adaptive, dt=dt, epsilon=epsilon)
    inference_time = time() - start
    w_dist = wasserstein(samples[-1], x_1)
    log = f'Time to Sample: {inference_time} | Wasserstein Distance: {w_dist}'
    fig, f = plot(samples)
    kde = kde_plot(samples[-1])
    disp = displacement_plot(samples)
    return log, fig, f, kde, disp

def eval(args):
    n_samples = 10000
    device = torch.device(f'cuda:{args.device}')

    test_x_0 = torch.tensor(inf_train_gen(args.source, batch_size=n_samples)).float().to(device)
    test_x_1 = torch.tensor(inf_train_gen(args.target, batch_size=n_samples)).float().to(device)

    name = get_folder_name(args)
    file_name = get_file_name(args)
    eps = [0.1]
    dts = [None]
    gamma = get_gamma(args.gamma)
    interpolant = get_interpolant(args.interpolant, gamma)
    model = SIModel(interpolant, args.noise).to(device)
    if os.path.exists(f'{name}/model.pt'):
        model.load_state_dict(torch.load(f'{name}/model.pt'))
        print('Model Loaded')
        with open(f'Plots/Toy/log.txt', 'a') as f:
            for ep in eps:
                for dt in dts:
                    log, fig_traj, fig, kde, disp = eval_step(model, test_x_0.to(device), test_x_1.to(device), dt is None, 1e-3 if dt is None else dt, epsilon=ep)
                    f.write(f'dt | {ep} | {file_name} | {log}\n')
                    print(f'dt | {ep} | {file_name} | {log}')
                    disp.savefig(f'{file_name}.pdf', bbox_inches='tight')
                    plt.close('all')
    else:
        print(f'Error | Directory not found')

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    eval(args)