#!/bin/bash

interpolant=Linear
gamma=Sqrt
coupling=Quad

i=0
for data in swissroll circles rings moons 8gaussians pinwheel 2spirals checkerboard line cos; do
    python si.py --data $data --device $((i%4)) &
    python si.py --data $data --device $((i%4)) --epsilon_t &
    i=i+1
done