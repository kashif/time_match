import numpy as np
import os
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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
parser.add_argument('--clip', type=float, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(f'cuda:{args.device}')
n_samples = 50000
bsz = 128
print(args)

classifier = nn.Sequential(
    nn.Linear(2, 128),
    nn.LayerNorm(128),
    nn.LeakyReLU(),
    nn.Linear(128, 128),
    nn.LayerNorm(128),
    nn.LeakyReLU(),
    nn.Linear(128, 128),
    nn.LayerNorm(128),
    nn.LeakyReLU(),
    nn.Linear(128, 1)
).to(device)
print(classifier)
optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.99)

def accuracy(pred, y):
    return 100. * ((pred > 0.5).long() == y.long()).float().mean()

if args.mode == 'SI':
    name = f'results/{args.mode}/{args.data}/{args.interpolant}_{args.gamma}/{args.seed}_{args.epsilon}'
    gamma = get_gamma(args.gamma)
    interpolant = get_interpolant(args.interpolant, gamma)
else:
    # name = f'results/{args.mode}/{args.data}/{args.interpolant}_{args.gamma}/{args.coupling}_{args.clip}/{args.seed}_{args.epsilon}'
    name = f'results/{args.mode}/{args.data}/{args.interpolant}_{args.gamma}/{args.clip}/{args.seed}_{args.epsilon}'
    gamma = get_gamma(args.gamma)
    interpolant = get_learned_interpolant(args.interpolant, args.coupling, gamma, hidden_dim=32, M=5).to(device)

source = Gaussian(2).to(device)
model = Model(interpolant).to(device)
model.load_state_dict(torch.load(f'{name}/model.pt'))
criterion = nn.BCEWithLogitsLoss()

# test_data = model.sample(source, 100, device)[-1].detach().cpu()

# sns.kdeplot(
#     x=test_data[:, 0],
#     y=test_data[:, 1],
#     fill=True, thresh=0, 
#     levels=100, cmap='viridis',
#     cut=8,
# )
# plt.gca().set_xlim(-4., 4.)
# plt.gca().set_ylim(-4., 4.)
# plt.savefig('trial.png')
# plt.close()

x_true = torch.tensor(inf_train_gen(args.data, batch_size=n_samples)).float()
x_gen = model.sample(source.sample(n_samples).to(device))[-1].detach().cpu()

fig, axs = plt.subplots(1,2, figsize=(10,4))
axs[0].scatter(x_true[:, 0], x_true[:, 1], s=1)
axs[1].scatter(x_gen[:, 0], x_gen[:, 1], s=1)
fig.savefig('gen.png')
plt.close()

print('Data Loaded')

def train():
    idx_0 = np.random.choice(n_samples, size=(bsz,), replace=False)
    idx_1 = np.random.choice(n_samples, size=(bsz,), replace=False)
    with torch.no_grad():
        x_t = x_true[idx_0].to(device)
        x_g = x_gen[idx_1].to(device)
        # x_true = torch.tensor(inf_train_gen(args.data, batch_size=128)).float().to(device)
        # x_gen = model.sample(source.sample(128).to(device))[-1]
        data = torch.cat([x_t, x_g], dim=0)
        labels = torch.cat([torch.ones_like(x_t[...,0]), torch.zeros_like(x_g[..., 0])], dim=0)

    pred = classifier(data).squeeze(-1)
    loss = criterion(pred, labels)
    acc = accuracy(pred, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), acc

for i in range(100000):
    loss, acc = train()
    if i % 1000 == 0:
        print(f'{i}: {loss} | {acc}')