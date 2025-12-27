model_path="../output/pretrain"
echo ${model_path}

python -m vllm.entrypoints.openai.api_server \
    --model ${model_path} \
    --served-model-name medical \
    --dtype=auto \
    --port 8801 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 2048 \
    -tp 1 &