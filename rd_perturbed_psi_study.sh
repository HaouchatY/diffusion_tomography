#!/bin/bash

for i in $(seq 0 19);
do
    taskset -c $((i)),$((i+24)) python -u rd_phantom_perturbed_psi_sxr_analysis.py $((i * 50)) $(((i+1) * 50)) &
done
