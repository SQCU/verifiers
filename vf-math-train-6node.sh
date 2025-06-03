#!/bin/bash
##vf-math-train-6node.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
# Run training script from verifiers/, with .venv active
#try running on 6node
CUDA_VISIBLE_DEVICES=3,4,5 accelerate launch --num-processes 3 --config-file configs/zero3.yaml verifiers/examples/soft_math_train.py