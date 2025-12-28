#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python fastapi_server.py \
  --base_model ../../model/Qwen/Qwen3-0.6B \
  --gpus 0 \
  --port 8801