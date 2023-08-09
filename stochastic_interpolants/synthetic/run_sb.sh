#!/bin/bash

source=8gaussians

for target in moons pinwheel cos; do
    python sb.py --source $source --target $target --device 0 --ot --iters 100001 &
    python sb.py --source $source --target $target --device 0 --iters 100001 &
    python sb.py --source $source --target $target --device 0 --ot --iters 100001 --noise laplace &
    python sb.py --source $source --target $target --device 0 --iters 100001 --noise laplace &

    python sb.py --source $source --target $target --device 0 --ot --iters 100001 --epsilon 1. &
    python sb.py --source $source --target $target --device 0 --iters 100001 --epsilon 1. &
    python sb.py --source $source --target $target --device 0 --ot --iters 100001 --noise laplace --epsilon 1. &
    python sb.py --source $source --target $target --device 0 --iters 100001 --noise laplace --epsilon 1. &

    python sb.py --source $source --target $target --device 0 --ot --iters 100001 --epsilon 0.1 &
    python sb.py --source $source --target $target --device 0 --iters 100001 --epsilon 0.1 &
    python sb.py --source $source --target $target --device 0 --ot --iters 100001 --noise laplace --epsilon 0.1 &
    python sb.py --source $source --target $target --device 0 --iters 100001 --noise laplace --epsilon 0.1 &
done
wait