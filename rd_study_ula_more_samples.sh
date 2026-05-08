#!/bin/bash

for i in $(seq 0 19);
do
    taskset -c $((i)),$((i+24)) python -u rd_phantom_analysis_ula_more_samples.py $((i * 50)) $(((i+1) * 50)) "pilatus" &
done

wait
