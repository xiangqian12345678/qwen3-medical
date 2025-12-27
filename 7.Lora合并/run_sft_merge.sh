python merge_peft_adapter.py \
    --base_model ../model/Qwen/Qwen3-0.6B \
    --tokenizer_name_or_path ../output/tokenizers_merge \
    --lora_model ../output/sft_adapter \
    --output_dir ../output/sft_merge