#!/bin/bash

source=8gaussians

for interpolant in Linear EncDec Trig; do
    for gamma in Sqrt Quad Trig; do
        for target in moons pinwheel cos; do
            for seed in {1..3}; do
                python si.py --interpolant $interpolant --gamma $gamma --source $source --target $target --device 0 --ot --iters 10001 --seed $seed &
                python si.py --interpolant $interpolant --gamma $gamma --source $source --target $target --device 1 --iters 10001 --seed $seed &
                python si.py --interpolant $interpolant --gamma $gamma --source $source --target $target --device 2 --ot --iters 10001 --noise laplace --seed $seed &
                python si.py --interpolant $interpolant --gamma $gamma --source $source --target $target --device 3 --iters 10001 --noise laplace --seed $seed &
            done
        done
        wait
    done
done