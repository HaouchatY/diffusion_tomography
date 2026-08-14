#!/bin/bash

python diffusion_dmpx_phantom_analysis.py

wait

python diffusion_hyperparam_tuning.py

wait