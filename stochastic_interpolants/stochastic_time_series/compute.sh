#!/bin/bash

for dataset in solar_nips electricity_nips exchange_rate_nips traffic_nips; do
    for model in si si_noise si_epsilon si_epsilon_noise; do
        for epsilon in 0.1 1.0 2.0; do
            python compute.py --dataset $dataset --estimator $model/Linear_Sqrt/$epsilon
        done
    done
done