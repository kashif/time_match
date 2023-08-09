import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import AR
from utils import set_seed, get_interpolant
from model import ConditionalStochasticInterpolant
from plot_utils import plot
import os

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--interpolant', type=str, default='EncDec')
args = parser.parse_args()

name = 'results/si_prev_cond'
if not os.path.exists(f'{name}/plots'):
    os.makedirs(f'{name}/plots')

def sample_batch(batch_size):
    x = dataset.get_batch(batch_size).to(device)
    ts = torch.tensor(np.random.choice(args.T-2, size=(batch_size,))).long().unsqueeze(-1).to(device)
    
    x_1 = x.gather(1, ts+1)
    x_0 = x.gather(1, ts)
    cond = x_0
    t = torch.rand(batch_size).unsqueeze(-1).to(x_1)

    return x_0, x_1, cond, t

def train_step():
    si_prev_cond.zero_grad()
    x_0, x_1, cond, t = sample_batch(args.batch_size)
    loss = si_prev_cond.train_step(x_0, x_1, t, cond)
    loss.backward()
    optimizer.step()
    return loss.item()

set_seed(0)
device = torch.device('cuda:1')

dataset = AR(args.T, std=args.noise)
interpolant = get_interpolant(args.interpolant)
si_prev_cond = ConditionalStochasticInterpolant(interpolant, 3).to(device)
optimizer = torch.optim.Adam(si_prev_cond.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for i in range(args.iters):
    l = train_step()
    if i % 1000 == 0:
        print(f'{i} | {l}')
        scheduler.step()
torch.save(si_prev_cond.state_dict(), f'{name}/model.pt')

set_seed(1)
vis_examples = 3
bsz = vis_examples ** 2
total_examples = 100
n_samples = 50
x = dataset.get_batch(total_examples).to(device)
original = x.detach().cpu()

with open(f'{name}/loss.txt', 'w') as f:
    # ODE Sampling
    
    with torch.no_grad():
        prediction = si_prev_cond.forecast_conditional_new(x, args.T, ode=True).detach().cpu()
        test_loss = (prediction.mean(dim=0) - original).abs().mean()
        print(f'ODE Test Loss: {test_loss}')
        f.write(f'ODE Test Loss: {test_loss}\n')
        plot(original[:bsz], prediction[:, :bsz], dataset.get_gt_pred, ode=True, name=f'{name}/plots/ode')

    # SDE Sampling

    with torch.no_grad():
        for epsilon in [0.01, 0.1, 0.25, 0.5, 1., 5.]:
            prediction = si_prev_cond.forecast_conditional_new(x, args.T, epsilon=epsilon, ode=False).detach().cpu()
            test_loss = (prediction.mean(dim=0) - original).abs().mean()
            print(f'SDE Test Loss with Epsilon {epsilon}: {test_loss}')
            f.write(f'SDE Test Loss with Epsilon {epsilon}: {test_loss}\n')
            plot(original[:bsz], prediction[:, :bsz], dataset.get_gt_pred, ode=False, name=f'{name}/plots/sde_{epsilon}')
