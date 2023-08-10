#!/bin/bash

seed=$1
i=0

for data in swissroll circles rings moons 8gaussians pinwheel 2spirals checkerboard line cos; do
    ./eval.sh $seed $((i % 4)) $data &
    ./eval.sh $seed $(((i+1) % 4)) $data &
    ./eval.sh $seed $(((i+2) % 4)) $data &
    i=$((i+1))
done