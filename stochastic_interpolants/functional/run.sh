#!/bin/bash

device=1

python ddpm.py --device $device &
python ddpm.py --gp --device $device &
wait

for source in gp linear quadratic; do
    python fm.py --device 1 --source $source &
    python si_dep.py --device 1 --source $source &
    python si_dep.py --device 1 --source $source --gp &
    python si_indep.py --device 3 --source $source &
    python si_indep.py --device 3 --source $source --gp &
    wait
done