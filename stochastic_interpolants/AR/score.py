import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import AR
from utils import set_seed, get_betas
from model import SGM
from plot_utils import plot
import os

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=1024)
args = parser.parse_args()

if not os.path.exists(f'results/sgm'):
    os.makedirs(f'results/sgm')

def sample_batch(batch_size):
    x = dataset.get_batch(batch_size).to(device)
    ts = torch.tensor(np.random.choice(args.T-2, size=(batch_size,))).long().unsqueeze(-1).to(device)
    
    x_1 = x.gather(1, ts+1)
    cond = x.gather(1, ts)
    t = (1 - 1e-5) * torch.rand(batch_size).unsqueeze(-1).to(x_1) + 1e-5

    return x_1, cond, t

def train_step():
    sgm.zero_grad()
    x_1, cond, t = sample_batch(args.batch_size)
    loss = sgm.train_step(x_1, cond, t)
    loss.backward()
    optimizer.step()
    return loss.item()

set_seed(0)
device = torch.device('cuda:0')

betas = get_betas(args.diffusion_steps).to(device)
alphas = torch.cumprod(1 - betas, dim=0)

dataset = AR(args.T, std=args.noise)
sgm = SGM(0.1, 1.).to(device)
optimizer = torch.optim.Adam(sgm.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for i in range(args.iters):
    l = train_step()
    if i % 1000 == 0:
        print(f'{i} | {l}')
        scheduler.step()
torch.save(sgm.state_dict(), f'results/sgm/model.pt')

set_seed(1)
vis_examples = 3
bsz = vis_examples ** 2
total_examples = 100
n_samples = 50
x = dataset.get_batch(total_examples).to(device)
original = x.detach().cpu()

with open(f'results/sgm/loss.txt', 'w') as f:
    # ODE Sampling
    
    with torch.no_grad():
        prediction = sgm.forecast(x, args.T, ode=True).detach().cpu()
        fig, axs = plt.subplots(nrows=vis_examples, ncols=vis_examples, figsize=(32, 32))
        test_loss = (prediction.mean(dim=0) - original).abs().mean()
        print(f'ODE Test Loss: {test_loss}')
        f.write(f'ODE Test Loss: {test_loss}\n')
        plot(original[:bsz], prediction[:, :bsz], axs, dataset.get_gt_pred, ode=True)
        fig.savefig('results/sgm/ode.png', dpi=fig.dpi)

    # SDE Sampling

    with torch.no_grad():
        prediction = sgm.forecast(x, args.T, ode=False).detach().cpu()
        fig, axs = plt.subplots(nrows=vis_examples, ncols=vis_examples, figsize=(32, 32))
        test_loss = (prediction.mean(dim=0) - original).abs().mean()
        print(f'SDE Test Loss: {test_loss}')
        f.write(f'SDE Test Loss: {test_loss}\n')
        plot(original[:bsz], prediction[:, :bsz], axs, dataset.get_gt_pred, ode=False)
        fig.savefig('results/sgm/sde.png', dpi=fig.dpi)