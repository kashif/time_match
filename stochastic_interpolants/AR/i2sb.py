import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import AR
from utils import set_seed, i2sb_betas, compute_gaussian_product_coef
from model import I2SB
from plot_utils import plot
from functools import partial
import os

parser = argparse.ArgumentParser()
parser.add_argument('--T', type=int, default=20)
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--beta_max', type=float, default=1e-4)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--ot', action='store_true', default=False)
args = parser.parse_args()

name = 'results/i2sb'
if args.ot:
    name = f'results/ot_i2sb'

if not os.path.exists(f'{name}/plots'):
    os.makedirs(f'{name}/plots')

def sample_batch(batch_size):
    x = dataset.get_batch(batch_size).to(device)
    ts = torch.tensor(np.random.choice(args.T-2, size=(batch_size,))).long().unsqueeze(-1).to(device)
    
    x_0 = x.gather(1, ts)
    x_1 = x.gather(1, ts+1)
    t = torch.randint(0, args.diffusion_steps, size=(batch_size, 1)).to(x_1)

    return x_0, x_1, t.long(), None

def train_step():
    i2sb.zero_grad()
    x_0, x_1, t, cond = sample_batch(args.batch_size)
    loss = i2sb.train_step(x_0, x_1, cond, t, ode=args.ot)
    loss.backward()
    optimizer.step()
    return loss.item()

set_seed(0)
device = torch.device('cuda:1')

betas = i2sb_betas(n_timestep=args.diffusion_steps, linear_end=args.beta_max)
betas = np.concatenate([betas[:args.diffusion_steps//2], np.flip(betas[:args.diffusion_steps//2])])
std_fwd = np.sqrt(np.cumsum(betas))
std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
std_sb = np.sqrt(var)

to_torch = partial(torch.tensor, dtype=torch.float32)
betas = to_torch(betas).to(device)
std_fwd = to_torch(std_fwd).to(device)
std_bwd = to_torch(std_bwd).to(device)
std_sb  = to_torch(std_sb).to(device)
mu_x0 = to_torch(mu_x0).to(device)
mu_x1 = to_torch(mu_x1).to(device)

dataset = AR(args.T, std=args.noise)
i2sb = I2SB(betas, std_fwd, std_bwd, std_sb, mu_x0, mu_x1, in_dim=2).to(device)
optimizer = torch.optim.Adam(i2sb.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for i in range(args.iters):
    l = train_step()
    if i % 1000 == 0:
        print(f'{i} | {l}')
        scheduler.step()
torch.save(i2sb.state_dict(), f'{name}/model.pt')

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
        prediction = i2sb.forecast(x, args.T, ode=True).detach().cpu()
        test_loss = (prediction.mean(dim=0) - original).abs().mean()
        print(f'ODE Test Loss: {test_loss}')
        f.write(f'ODE Test Loss: {test_loss}\n')
        plot(original[:bsz], prediction[:, :bsz], dataset.get_gt_pred, ode=True, name=f'{name}/plots/ode')

    # SDE Sampling

    with torch.no_grad():
        prediction = i2sb.forecast(x, args.T, ode=False).detach().cpu()
        test_loss = (prediction.mean(dim=0) - original).abs().mean()
        print(f'SDE Test Loss: {test_loss}')
        f.write(f'SDE Test Loss: {test_loss}\n')
        plot(original[:bsz], prediction[:, :bsz], dataset.get_gt_pred, ode=False, name=f'{name}/plots/sde')
