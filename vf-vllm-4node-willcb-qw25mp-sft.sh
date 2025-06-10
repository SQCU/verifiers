#!/bin/bash
##vf-vllm-4node-willcb-qw25mp-sft.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
export WANDB_DISABLED=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model "willcb/Qwen2.5-7B-Math-Python-SFT" \
    --tensor-parallel-size 2 --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000
#CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_python_4node.py