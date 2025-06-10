#!/bin/bash
##vf-vllm-6node-host-qw3-06b.sh
source .venv/bin/activate
export OPENAI_API_KEY=FRICK0AWAY0AND0BEGONE
# Launch vLLM inference server from verifiers/, with .venv active
#https://huggingface.co/Qwen/Qwen3-0.6B-Base
#tp=1 bc its 0.6b.
#try running on 6node
CUDA_VISIBLE_DEVICES=0,1,2 vf-vllm \
--model "Qwen/Qwen3-0.6B-Base" \
--tensor-parallel-size 1 --max-model-len 8192 \
--gpu-memory-utilization 0.9 --enable-prefix-caching \
--dtype bfloat16 \
--host 0.0.0.0 --port 8000