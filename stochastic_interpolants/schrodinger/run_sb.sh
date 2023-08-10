#!/bin/bash

interpolant=Linear
gamma=Sqrt
coupling=Quad

i=0
for data in swissroll circles rings moons 8gaussians pinwheel 2spirals checkerboard line cos; do
    for clip in None 0. 0.0001 0.01 0.1; do
        python sb.py --gamma $gamma --interpolant $interpolant --data $data --coupling $coupling --device $((i%4)) &
        python sb.py --gamma $gamma --interpolant $interpolant --data $data --coupling $coupling --device $((i%4)) --epsilon_t &
        i=i+1
    done
done