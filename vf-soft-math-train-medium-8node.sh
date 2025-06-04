#!/bin/bash
##vf-soft-math-train-medium-8node.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
export WANDB_DISABLED=true
# Run training script from verifiers/, with .venv active
#try running on 4node
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file configs/zero3.yaml verifiers/examples/soft_math_train_medium.py