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
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='solar_nips')
parser.add_argument('--estimator', type=str, default='ddpm')
parser.add_argument('--gamma', type=str, default='Sqrt')
parser.add_argument('--interpolant', type=str, default='Linear')
parser.add_argument('--epsilon', type=float, default=2.)
parser.add_argument('--beta_start', type=float, default=1e-4)
parser.add_argument('--beta_end', type=float, default=2e-2)
parser.add_argument('--steps', type=int, default=150)
parser.add_argument('--dt', type=float, default=None)
parser.add_argument('--sde_solver', type=str, default='manual')
parser.add_argument('--start_noise', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

print(args)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(args.seed)
name = f'results/{args.dataset}/{args.estimator}'
if args.start_noise and args.estimator in ['i2sb', 'si', 'si_epsilon', 'si_reg', 'si_epsilon_reg', 'sfm']:
    name = f'{name}_noise'
if args.estimator in ['si', 'si_epsilon', 'si_reg', 'si_epsilon_reg']:
    name = f'{name}/{args.interpolant}_{args.gamma}/{args.epsilon}'

name = f'{name}/{args.seed}'
if not os.path.exists(name):
    os.makedirs(name)

dataset = get_dataset(args.dataset, regenerate=False)

if 'wiki' in args.dataset:
    max_target_dim = 2000
else:
    max_target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

train_grouper = MultivariateGrouper(
    max_target_dim=int(max_target_dim)
)

test_grouper = MultivariateGrouper(
    num_test_dates=int(len(dataset.test) / len(dataset.train)),
    max_target_dim=int(max_target_dim),
)
dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
evaluator = MultivariateEvaluator(
    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
)

estimator = get_estimator(args, dataset)

predictor = estimator.train(dataset_train, cache_data=True, shuffle_buffer_length=1024)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset_test, predictor=predictor, num_samples=100
)
forecasts = list(forecast_it)
targets = list(ts_it)
agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

log = get_log(agg_metric)
with open(f'{name}/log.txt', 'w') as f:
    f.write(log)
print(log)

plot(
    target=targets[0],
    forecast=forecasts[0],
    prediction_length=dataset.metadata.prediction_length,
)
plt.savefig(f'{name}/results.png')
