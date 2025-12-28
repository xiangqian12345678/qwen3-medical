#关键参数说明:
#--teacher_model_path: 教师模型的路径（Qwen3-4B）。
#--student_model_path: 学生模型的路径（Qwen3-1.7B）。
#--output_dir: 训练产出的LoRA适配器将被保存在这里，它属于学生模型。
#--alpha: 蒸馏损失的权重。值越高，学生模型越倾向于模仿老师；值越低，越倾向于拟合标准答案。0.5到0.9是常用范围。
#--temperature: 蒸馏温度。用于平滑教师模型的输出，让学生能学到更"软"的知识。通常大于1，例如2.0或2.5。

python distill.py \
    --teacher_model_path ../model/Qwen/Qwen3-1.7B \
    --student_model_path ../model/Qwen/Qwen3-0.6B \
    --dataset_path ../data/distill/dirty_chinese_dpo.json \
    --output_dir ../output/distilled_adapter \
    --alpha 0.7 \
    --temperature 1