#!/bin/bash
##vf-math-train-6node.sh
source .venv/bin/activate
# Run training script from verifiers/, with .venv active
#try running on 6node
CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/soft_math_train.py