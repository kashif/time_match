#!/bin/bash

i=0
for dataset in wiki-rolling_nips; do
    for seed in {8..10}; do
        python main.py --dataset $dataset --estimator si_epsilon --seed $seed --epsilon 1.0 --device 2 &
        python main.py --dataset $dataset --estimator si_epsilon --start_noise --seed $seed --epsilon 1.0 --device 3 &
        wait
    done
done