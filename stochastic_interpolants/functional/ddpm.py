import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from data import *
from interpolant import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--gp', action='store_true', default=False)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

batch_size = 256
min_T = 32
max_T = 128
eval_T = 100
iters = 10000

name = 'results/ddpm'
device = torch.device(f'cuda:{args.device}')
if args.gp:
    name = f'{name}_gp'

if not os.path.exists(name):
    os.makedirs(name)

def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0)

dataset_ts = torch.rand(10, eval_T, 1).sort(1)[0].to(device)
dataset_x1 = get_sin(dataset_ts)

for i in range(10):
    plt.plot(dataset_ts[i,:,0].cpu().numpy(), dataset_x1[i,:,0].cpu().numpy(), color='C0', alpha=1 / (i + 1))
plt.title('10 samples from the dataset\nEach curve is one "data point"')
plt.xlabel('t')
plt.ylabel('x')
plt.savefig(f'{name}/data.png')
plt.close()

def train_step(batch_size):
    T = min_T + np.random.choice(max_T - min_T + 1)
    ts = torch.rand(batch_size, T, 1).to(device)
    t = torch.rand(ts.shape[0]).view(-1, 1, 1).repeat(1, T, 1).to(device)
    x_0 = get_data(ts, 'gp').to(device)
    x_1 = get_data(ts, 'sin').to(device)

    ts = torch.rand(batch_size, T, 1).to(device)
    t = torch.randint(0, diffusion_steps, size=(ts.shape[0],)).view(-1, 1, 1).repeat(1, T, 1).to(device)
    x_1 = get_sin(ts).to(device)

    model.zero_grad()
    loss = model.train_step(x_1, ts, t).mean()
    loss.backward()
    optim.step()
    return loss.item()

model = DDPM(alphas, betas, diffusion_steps, gp=args.gp).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

pbar = tqdm(range(iters + 1))
for i in pbar:
    loss = train_step(batch_size)
    if i % 100 == 0:
        pbar.set_description("Loss %s" % loss)

    if i % 2500 == 0:
        num_samples = 10
        ts = torch.rand(num_samples, eval_T, 1).sort(1)[0].to(device)
        x_1 = model.sample(ts).view(num_samples, eval_T, 1)

        for j in range(10):
            plt.plot(ts[j,:,0].cpu().numpy(), x_1[j,:,0].cpu().numpy(), color='C0', alpha=1 / (j + 1))
        plt.savefig(f'{name}/model_{i}.png')
        plt.close()