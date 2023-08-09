#!/bin/bash

dataset=$1
estimator=$2
extra=$3

i=0
for interpolant in Linear Trig EncDec; do
    for gamma in Sqrt Quad Trig; do
        for epsilon in 0.1 1.0 2.0; do
            for seed in {0..2}; do
                python main.py --dataset $dataset --estimator $estimator --interpolant $interpolant --gamma $gamma --epsilon $epsilon --device $((i%2)) --seed $seed $extra &
                i=$((i+1))
            done
        done
        wait
    done
done

# for dataset in electricity_nips exchange_rate_nips taxi_30min traffic_nips wiki-rolling_nips; do
