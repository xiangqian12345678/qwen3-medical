#!/bin/bash

# 如果执行失败，注意修正脚本格式
# 命令：  sed -i 's/\r$//' request_vllm.sh
curl http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medical",
    "messages": [
      {
        "role": "system",
        "content": "you are a helpful assistant."
      },
      {
        "role": "user",
        "content": "please talk about Beijing."
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
  }'
