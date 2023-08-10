#!/bin/bash

seed=$1
device=$2
data=$3

for gamma in Trig Quad Sqrt; do
    for interpolant in Linear Trig EncDec; do
        python mmd.py --mode SI --gamma $gamma --interpolant $interpolant --data $data --seed $seed --device $device
        for coupling in Quad Fourier; do
            python mmd.py --mode SB --gamma $gamma --interpolant $interpolant --data $data --seed $seed --coupling $coupling --device $device
            python mmd.py --mode SI_Static --gamma $gamma --interpolant $interpolant --data $data --seed $seed --coupling $coupling --device $device
        done
    done
done