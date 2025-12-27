echo '-----------------开始计算Qwen3-0.6B-4bit的困惑度-----------------------'
python perplexity_valuate.py \
  --bnb_path ../output/quant/Qwen3-0.6B-4bit \
  --data_path ../data/finetune/medical_sft_1K_format.jsonl


echo '-----------------开始计算Qwen3-0.6B的困惑度-----------------------'
python perplexity_valuate.py \
  --bnb_path ../model/Qwen/Qwen3-0.6B \
  --data_path ../data/finetune/medical_sft_1K_format.jsonl
