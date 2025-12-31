import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from loguru import logger


def load_processed_datasets(cache_dir):
    train_dataset = load_from_disk(os.path.join(cache_dir, "processed_datasets", "train"))
    val_dataset = load_from_disk(os.path.join(cache_dir, "processed_datasets", "validation"))
    return train_dataset, val_dataset


def train(model_name, tokenizer_name, cache_dir, output_dir, epochs=3, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataset, val_dataset = load_processed_datasets(cache_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型训练完成并保存到 {output_dir}")


if __name__ == "__main__":
    train(
        model_name="../../model/Qwen/Qwen3-0.6B",
        tokenizer_name="../../output/tokenizers_merge",
        cache_dir="../../output/cache",
        output_dir="../../output/medical",
        epochs=3,
        batch_size=2
    )
