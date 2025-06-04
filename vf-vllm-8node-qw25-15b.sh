#!/bin/bash
##vf-vllm-8node-qw25-15b.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
export WANDB_DISABLED=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# Launch vLLM inference server from verifiers/, with .venv active
#https://huggingface.co/Qwen/Qwen2.5-Math-1.5B
#tp=1 bc its 0.6b.
#try running on 6node
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm \
--model "Qwen/Qwen2.5-Math-1.5B" \
--tensor-parallel-size 2 --max-model-len 8192 \
--gpu-memory-utilization 0.9 --enable-prefix-caching \
--dtype bfloat16 \
--host 0.0.0.0 --port 8000