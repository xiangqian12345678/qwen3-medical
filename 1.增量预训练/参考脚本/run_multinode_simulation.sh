#!/bin/bash

# 模拟多节点分布式训练
# 在同一台机器上启动两个不同的进程来模拟不同节点

# 节点0 (Master)
NODE_RANK=0 CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nnodes 2 \
    --nproc_per_node 2 \
    --node_rank $NODE_RANK \
    --master_addr "127.0.0.1" \
    --master_port 29502 \
    --rdzv_id "multinode_sim" \
    --rdzv_backend "c10d" \
    --rdzv_endpoint "127.0.0.1:29502" \
    pretraining.py \
    --model_name_or_path ../model/Qwen/Qwen3-0.6B \
    --train_file_dir ../data/pretrain \
    --validation_file_dir ../data/pretrain \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 0.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir ../output/pretrain_multinode_node0 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ../output/cache &

# 节点1 (Worker)
sleep 2  # 等待master节点启动
NODE_RANK=1 CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nnodes 2 \
    --nproc_per_node 2 \
    --node_rank $NODE_RANK \
    --master_addr "127.0.0.1" \
    --master_port 29502 \
    --rdzv_id "multinode_sim" \
    --rdzv_backend "c10d" \
    --rdzv_endpoint "127.0.0.1:29502" \
    pretraining.py \
    --model_name_or_path ../model/Qwen/Qwen3-0.6B \
    --train_file_dir ../data/pretrain \
    --validation_file_dir ../data/pretrain \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 0.5 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 10 \
    --block_size 512 \
    --group_by_length True \
    --output_dir ../output/pretrain_multinode_node1 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dtype bfloat16 \
    --bf16 \
    --device_map auto \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ../output/cache

wait  # 等待所有后台进程完成