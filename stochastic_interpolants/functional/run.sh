#!/bin/bash

device=1

# python ddpm.py --device $device &
# python ddpm.py --gp --device $device &
# wait

for source in gp linear quadratic; do
#     python fm.py --device $device --source $source --gp &
# done
# wait
#     i=0
    for interpolant in Linear EncDec Trig; do
        # for gamma in Zero Sqrt Quad Trig; do
        python si_indep.py --device 1 --source $source --interpolant $interpolant --gamma Zero --gp &
        python si_indep.py --device 1 --source $source --interpolant $interpolant --gamma Sqrt --gp  &
        python si_indep.py --device 2 --source $source --interpolant $interpolant --gamma Quad --gp  &
        python si_indep.py --device 2 --source $source --interpolant $interpolant --gamma Trig --gp  &
        # python si_indep.py --device $((1+i%2)) --source $source --interpolant $interpolant --gamma $gamma &
        wait
    done
done