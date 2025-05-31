#!/bin/bash
##uv-vllm-host-qw3-0.6b-b.sh
source .venv/bin/activate
# Launch vLLM inference server from verifiers/, with .venv active
#https://huggingface.co/Qwen/Qwen3-0.6B-Base
#tp=1 bc its 0.6b. 
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py \
--model "Qwen/Qwen3-0.6B-Base" \
--tensor_parallel_size 1 --max_model_len 8192 \
--gpu_memory_utilization 0.9 --enable_prefix_caching True \
--dtype bfloat16 \
--host 0.0.0.0 --port 8000