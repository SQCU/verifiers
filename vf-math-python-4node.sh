#!/bin/bash
##vf-math-python-4node.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
#export WANDB_DISABLED=true
export WANDB_OFFLINE=true
#...

#CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model "willcb/Qwen2.5-7B-Math-Python-SFT" \
#    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
#    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
#    --host 0.0.0.0 --port 8000
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_python_4node.py