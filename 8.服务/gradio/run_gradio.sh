#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python gradio_server1.py \
  --base_model ../../model/Qwen/Qwen3-0.6B \
  --lora_model "" \
  --template_name "qwen" \
  --system_prompt "you are a helpfull assistant" \
  --context_len 2048 \
  --max_new_tokens 512 \
  --port 8801 \
  --share