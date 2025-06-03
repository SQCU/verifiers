#!/bin/bash
##vf-math-train-4node.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
# Run training script from verifiers/, with .venv active
#try running on 4node
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/soft_math_train.py