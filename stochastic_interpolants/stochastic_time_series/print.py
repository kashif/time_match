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

estimators = ['ddpm', 'sgm', 'fm', 'i2sb_noise', 'i2sb', 'si_noise/Linear_Sqrt/1.0', 'si/Linear_Sqrt/1.0', 'si_epsilon_noise/Linear_Sqrt/1.0', 'si_epsilon/Linear_Sqrt/1.0']
datasets = ['solar_nips', 'electricity_nips', 'exchange_rate_nips', 'traffic_nips', 'taxi_30min', 'wiki-rolling_nips']

def get_metric(idx, dataset, estimator):
    arr = []
    for seed in range(1, 11):
        with open(f'results/{dataset}/{estimator}/{seed}/log.txt', 'r') as f:
            data = f.read().split('\n')
        
        arr.append(float(data[idx].split(':')[-1]))
    
    arr = np.array(arr)
    return np.mean(arr), np.std(arr)

idx = 5
for estimator in estimators:
    line = f'{estimator} '
    for dataset in datasets:
        mean, std = get_metric(idx, dataset, estimator)
        line += f'& \\g{{{mean:.3f}}}{{{std:.3f}}} '
    print(line + '\\\\')