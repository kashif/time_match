import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from data import *
from interpolant import *
from gamma import *
from model import *

from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--source', type=str, default='gp')
parser.add_argument('--target', type=str, default='sin')
args = parser.parse_args()

batch_size = 256
min_T = 32
max_T = 128
eval_T = 100
iters = 25000

name = 'results/fm'
device = torch.device(f'cuda:{args.device}')
name = f'{name}/{args.source}_{args.target}'

if not os.path.exists(name):
    os.makedirs(name)

dataset_ts = torch.rand(10, eval_T, 1).sort(1)[0].to(device)
dataset_x1 = get_data(dataset_ts, args.target)

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
    x_0 = get_data(ts, args.source).to(device)
    x_1 = get_data(ts, args.target).to(device)

    model.zero_grad()
    loss = model(x_0, x_1, t, ts).mean()
    loss.backward()
    optim.step()
    return loss.item()

dataset_ts = torch.rand(10, eval_T, 1).sort(1)[0].to(device)
dataset_x0 = get_data(dataset_ts, args.source).to(device)
dataset_x1 = get_data(dataset_ts, args.target).to(device)

model = FlowMatching().to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

pbar = tqdm(range(iters + 1))
for i in pbar:
    loss = train_step(batch_size)
    if i % 100 == 0:
        pbar.set_description("Loss %s" % loss)

    if i % 5000 == 0:
        num_samples = 10
        ts = torch.rand(num_samples, eval_T, 1).sort(1)[0].to(device)
        x_0 = get_data(dataset_ts, args.source).to(device)
        x_1 = model.sample(x_0, ts).view(num_samples, eval_T, 1)

        for j in range(10):
            plt.plot(ts[j,:,0].cpu().numpy(), x_1[j,:,0].cpu().numpy(), color='C0', alpha=1 / (j + 1))
        plt.savefig(f'{name}/sample_{i}.png')
        plt.close()