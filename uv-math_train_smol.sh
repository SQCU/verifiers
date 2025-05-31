#!/bin/bash
##uv-math_train.sh
source .venv/bin/activate
# Run training script from verifiers/, with .venv active
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file configs/zero3.yaml verifiers/examples/math_train_qw3_0.6b.py