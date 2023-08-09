#!/bin/bash

for dataset in solar_nips electricity_nips exchange_rate_nips traffic_nips taxi_30min wiki-rolling_nips; do
    python compute.py --dataset $dataset --estimator ddpm
    python compute.py --dataset $dataset --estimator sgm
    python compute.py --dataset $dataset --estimator fm
    python compute.py --dataset $dataset --estimator i2sb
    python compute.py --dataset $dataset --estimator i2sb_noise

    for model in si si_noise si_epsilon si_epsilon_noise; do
        for epsilon in 1.0; do
            python compute.py --dataset $dataset --estimator $model/Linear_Sqrt/$epsilon
        done
    done
done