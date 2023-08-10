import numpy as np
import os
import argparse
from time import time

import seaborn as sns
import matplotlib.pyplot as plt

import torch
from data import inf_train_gen
from models import Model, EpsilonModel
from interpolants import *
from utils import *
from distributions import *
from metrics import wasserstein

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=str, default='Quad')
    parser.add_argument('--interpolant', type=str, default='EncDec')
    parser.add_argument('--data', type=str, default='cos')
    parser.add_argument('--epsilon', type=float, default=1.)
    parser.add_argument('--epsilon_t', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iters', type=int, default=10001)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    print(args)
    return args

def get_folder_name(args):
    name = f'results/SI'
    if args.epsilon_t:
        name = f'{name}_Epsilon'
    name = f'{name}/{args.data}/{args.interpolant}_{args.gamma}/{args.seed}_{args.epsilon}'
    return name
    
def train_step(model, optimizer, source, target, batch_size, device):
    x_0, x_1, t = get_data(source, target, batch_size, device)    
    optimizer.zero_grad()
    l = model.train_step(x_0, x_1, t).mean()
    l.backward()
    optimizer.step()
    return l.item()

def get_interpolations(model, source, target, ts, n_samples, device):
    x_0 = source.sample(n_samples).to(device)
    x_1 = torch.tensor(inf_train_gen(target, batch_size=n_samples)).float().to(device)
    interps = []
    for t in ts:
        interps.append(model.interpolant.interpolate(x_0, x_1, torch.ones_like(x_0[..., :1]) * t)[0])
    interps = torch.stack(interps)
    fig = plot(interps, (20, 3))
    return fig

def eval_step(model, source, target, n_samples, device, adaptive=True, dt=1e-3):
    x_0 = source.sample(n_samples).to(device)
    x_1 = torch.tensor(inf_train_gen(target, batch_size=n_samples)).float().to(device)
    start = time()
    samples = model.sample(x_0, adaptive=adaptive, dt=dt)
    inference_time = time() - start
    w_dist = wasserstein(samples[-1], x_1)
    log = f'Time to Sample: {inference_time} | Wasserstein Distance: {w_dist}'
    fig = plot(samples)
    return log, fig

def train(args):
    device = torch.device(f'cuda:{args.device}')
    batch_size = 1024
    max_steps = args.iters
    n_samples = 5000
    interpolant_ts = [0., 0.25, 0.5, 0.8, 0.9, 0.95, 0.999, 1.]

    target = inf_train_gen(args.data, batch_size=5000)
    target = torch.tensor(target).float()
    source = Gaussian(2).to(device)
    source_sample = source.sample(batch_size).detach().cpu()

    name = get_folder_name(args)
    if not os.path.exists(name):
        os.makedirs(name)

    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].scatter(source_sample[:, 0], source_sample[:, 1])
    axs[1].scatter(target[:, 0], target[:, 1])
    plt.savefig(f'{name}/data.png')
    plt.close()

    gamma = get_gamma(args.gamma)
    interpolant = get_interpolant(args.interpolant, gamma)
    if args.epsilon_t:
        model = EpsilonModel(interpolant, args.epsilon).to(device)
    else:
        model = Model(interpolant, args.epsilon).to(device)
    optimizer = torch.optim.Adam(model.model.parameters())

    print(model)
    with open(f'{name}/log.txt', 'w') as f:
        for i in range(max_steps):
            l = train_step(model, optimizer, source, target, batch_size, device)
            if i % 1000 == 0:
                print(f'{i} | {l}')
                f.write(f'{i} | {l}\n')

        torch.save(model.state_dict(), f'{name}/model.pt')
        log, fig = eval_step(model, source, args.data, n_samples, device)
        print(log)
        f.write(log)
        fig.savefig(f'{name}/samples.png')
        fig = get_interpolations(model, source, args.data, interpolant_ts, n_samples, device)
        fig.savefig(f'{name}/interpolations.png')

def eval(args):
    n_samples = 5000
    device = torch.device(f'cuda:{args.device}')

    source = Gaussian(2).to(device)
    source_sample = source.sample(n_samples).detach().cpu()

    name = get_folder_name(args)
    dts = [None, 1e-3, 1e-2, 1e-1]
    gamma = get_gamma(args.gamma)
    interpolant = get_interpolant(args.interpolant, gamma)
    model = Model(interpolant, args.epsilon).to(device)
    if os.path.exists(f'{name}/model.pt'):
        model.load_state_dict(torch.load(f'{name}/model.pt'))
        print('Model Loaded')
        with open(f'{name}/results.txt', 'w') as f:
            for dt in dts:
                if dt is None:
                    file_name = 'Adaptive'
                else:
                    file_name = f'{dt}'
                log, fig = eval_step(model, source, args.data, n_samples, device, dt is None, 1e-3 if dt is None else dt)
                f.write(f'dt: {file_name} | {log}')
                print(f'dt: {file_name} | {log}')
                fig.savefig(f'{name}/{file_name}.png')
    else:
        print(f'Error | Directory not found')

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    if args.eval:
        eval(args)
    else:
        train(args)