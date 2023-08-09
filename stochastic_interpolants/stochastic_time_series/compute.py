import random
import numpy as np
import pandas as pd
import os 

import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from pts.dataset.repository.datasets import dataset_recipes
from utils import *
import argparse

# ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='solar_nips')
parser.add_argument('--estimator', type=str, default='ddpm')
args = parser.parse_args()

print(f'{args.dataset} | {args.estimator}')
crps = []
crps_sum = []
for seed in range(1, 11):
    if not os.path.exists(f'results/{args.dataset}/{args.estimator}/{seed}/log.txt'):
        print(f'results/{args.dataset}/{args.estimator}/{seed}/log.txt')
        continue
    with open(f'results/{args.dataset}/{args.estimator}/{seed}/log.txt', 'r') as f:
        data = f.read().split('\n')
    
    crps.append(float(data[0].split(':')[-1]))
    crps_sum.append(float(data[5].split(':')[-1]))

crps = np.array(crps)
crps_sum = np.array(crps_sum)

print(f'{np.mean(crps):.3f} | {np.std(crps):.3f} | {len(crps)}')
print(f'{np.mean(crps_sum):.3f} | {np.std(crps_sum):.3f} | {len(crps_sum)}')