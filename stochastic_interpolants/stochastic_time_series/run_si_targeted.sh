#!/bin/bash

# for seed in {0..2}; do
#     for dataset in electricity_nips; do
#         for epsilon in 0.1 1.0 2.0; do
#             python main.py --dataset $dataset --estimator si --device 0 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon --start_noise &
#             python main.py --dataset $dataset --estimator si --device 0 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon &
#             python main.py --dataset $dataset --estimator si_epsilon --device 1 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon --start_noise &
#             python main.py --dataset $dataset --estimator si_epsilon --device 1 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon &
#             python main.py --dataset $dataset --estimator si_reg --device 2 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon --start_noise &
#             python main.py --dataset $dataset --estimator si_reg --device 2 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon &
#             python main.py --dataset $dataset --estimator si_epsilon_reg --device 3 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon --start_noise &
#             python main.py --dataset $dataset --estimator si_epsilon_reg --device 3 --seed $seed --interpolant Linear --gamma Sqrt --epsilon $epsilon &
#         done
#         wait
#     done
# done

for seed in {0..2}; do
    for linear_start in 1e-4 1e-3 1e-2 1e-1 1.; do
        for linear_end in 5. 10. 20.; do
            python main.py --dataset solar_nips --beta_start $linear_start --beta_end $linear_end --estimator sgm --device $((i%4)) --seed $seed &
            i=$((i+1))
        done
    done
    wait
done