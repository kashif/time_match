import numpy as np
import os
import argparse
from time import time

import matplotlib.pyplot as plt

import torch
from data import inf_train_gen
from models import SBModel
from interpolants import *
from utils import *
from distributions import *
from metrics import wasserstein

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=str, default='Sqrt')
    parser.add_argument('--interpolant', type=str, default='Linear')
    parser.add_argument('--coupling', type=str, default='Quad')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--fourier', type=int, default=5)
    parser.add_argument('--source', type=str, default='8gaussians')
    parser.add_argument('--target', type=str, default='moons')
    parser.add_argument('--epsilon', type=float, default=1.)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--noise', type=str, default='gaussian')
    parser.add_argument('--ot', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=10001)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    print(args)
    return args

def get_folder_name(args):
    name = f'results/SB'
    if args.ot:
        name = f'{name}_ot'
    name = f'{name}/{args.source}_{args.target}/{args.interpolant}_{args.gamma}/{args.noise}/{args.coupling}_{args.clip}_{args.epsilon}/{args.seed}'
    return name

def train_interpolant_step(model, optimizer, source, target, batch_size, device):
    x_0, x_1, t = get_data(source, target, batch_size, device)
    if args.ot:
        x_0, x_1 = get_ot_sample(x_0, x_1)
    model.zero_grad()
    l = -model.train_step(x_0, x_1, t).mean()
    l.backward()
    optimizer.step()
    if args.clip is not None:
        with torch.no_grad():
            for param in model.interpolant.parameters():
                param.clamp_(-args.clip, args.clip)

    return l.item()

def train_step(model, optimizer, source, target, batch_size, device):
    x_0, x_1, t = get_data(source, target, batch_size, device)
    if args.ot:
        x_0, x_1 = get_ot_sample(x_0, x_1)
    optimizer.zero_grad()
    l = model.train_step(x_0, x_1, t).mean()
    l.backward()
    optimizer.step()
    return l.item()

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

def train(args):
    device = torch.device(f'cuda:{args.device}')
    batch_size = 256
    max_steps = args.iters
    n_samples = 5000
    interpolant_ts = [0., 0.25, 0.5, 0.8, 0.9, 0.95, 0.999, 1.]

    test_x_0 = torch.tensor(inf_train_gen(args.source, batch_size=n_samples)).float().detach().cpu()
    test_x_1 = torch.tensor(inf_train_gen(args.target, batch_size=n_samples)).float().detach().cpu()

    source = torch.tensor(inf_train_gen(args.source, batch_size=10000)).float().detach().cpu()
    target = torch.tensor(inf_train_gen(args.target, batch_size=10000)).float().detach().cpu()

    name = get_folder_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].scatter(source[:, 0], source[:, 1], s=2)
    axs[1].scatter(target[:, 0], target[:, 1], s=2)
    plt.savefig(f'{name}/data.png')
    plt.close()

    gamma = get_gamma(args.gamma)
    interpolant = get_learned_interpolant(args.interpolant, args.coupling, gamma, hidden_dim=args.hidden_dim, M=args.fourier).to(device)
    model = SBModel(interpolant, args.noise, args.epsilon).to(device)
    optimizer_model = torch.optim.Adam(model.model.parameters())
    optimizer_interpolant = torch.optim.Adam(model.interpolant.parameters())

    print(model)
    with open(f'{name}/log.txt', 'w') as f:
        for i in range(max_steps):
            l = train_step(model, optimizer_model, source, target, batch_size, device)
            if i < max_steps - 10001 and i % 100 == 0:
                _ = train_interpolant_step(model, optimizer_interpolant, source, target, batch_size, device)
            if i % 1000 == 0:
                print(f'{i} | {l}')
                f.write(f'{i} | {l}\n')

        torch.save(model.state_dict(), f'{name}/model.pt')
        log, fig_traj, fig, kde, disp = eval_step(model, test_x_0.to(device), test_x_1.to(device), device)
        fig_interpolant = get_interpolations(model, test_x_0.to(device), test_x_1.to(device), interpolant_ts, device)
        print(log)
        f.write(log)
        fig_traj.savefig(f'{name}/sample_traj.png', bbox_inches='tight')
        fig.savefig(f'{name}/samples.png', bbox_inches='tight')
        kde.savefig(f'{name}/kde.png', bbox_inches='tight')
        disp.savefig(f'{name}/disp.png', bbox_inches='tight')
        fig_interpolant.savefig(f'{name}/interpolations.png', bbox_inches='tight')
        plt.close()

def eval(args):
    n_samples = 10000
    device = torch.device(f'cuda:{args.device}')

    test_x_0 = torch.tensor(inf_train_gen(args.source, batch_size=n_samples)).float().to(device)
    test_x_1 = torch.tensor(inf_train_gen(args.target, batch_size=n_samples)).float().to(device)

    name = get_folder_name(args)
    eps = [0., 0.1, 1.]
    dts = [None, 1e-2, 1e-1, 0.25, 0.5]
    gamma = get_gamma(args.gamma)
    interpolant = get_learned_interpolant(args.interpolant, args.coupling, gamma, hidden_dim=args.hidden_dim, M=args.fourier).to(device)
    model = SBModel(interpolant, args.noise, args.epsilon).to(device)
    if os.path.exists(f'{name}/model.pt'):
        model.load_state_dict(torch.load(f'{name}/model.pt'))
        print('Model Loaded')
        with open(f'{name}/results.txt', 'w') as f:
            for ep in eps:
                for dt in dts:
                    if dt is None:
                        file_name = f'Adaptive_{ep}'
                    else:
                        file_name = f'{dt}_{ep}'

                    log, fig_traj, fig, kde, disp = eval_step(model, test_x_0.to(device), test_x_1.to(device), dt is None, 1e-3 if dt is None else dt, epsilon=ep)
                    f.write(f'dt | {ep} | {file_name} | {log}\n')
                    print(f'dt | {ep} | {file_name} | {log}')
                    if not os.path.exists(f'{name}/{file_name}'):
                        os.makedirs(f'{name}/{file_name}')
                    fig_traj.savefig(f'{name}/{file_name}/sample_traj.png', bbox_inches='tight')
                    fig.savefig(f'{name}/{file_name}/samples.png', bbox_inches='tight')
                    kde.savefig(f'{name}/{file_name}/kde.png', bbox_inches='tight')
                    disp.savefig(f'{name}/{file_name}/disp.png', bbox_inches='tight')
                    plt.close()
    else:
        print(f'Error | Directory not found')

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    if args.eval:
        eval(args)
    else:
        train(args)