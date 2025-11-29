''' 
This script fine-tunes a language model for summarizing Chinese patent texts using supervised fine-tuning (SFT) with LoRA.
'''

import os
import random
import torch
import pandas as pd

from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
# from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from data_utils import set_seed, load_splits_as_datasets


# ===== model config =====
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "../models/sft"
MAX_LENGTH = 1024
SEED = 42

PROMPT_TEMPLATE = "请根据以下专利文本生成摘要:\n{text}\n\n摘要:"


def prepare_tokenizer_and_model():
    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model


def tokenize_function(example, tokenizer):
    text = str(example["text"])
    summary = str(example["summary"])

    prompt = PROMPT_TEMPLATE.format(text=text)
    target = summary.strip()

    full_text = prompt + " " + target

    tok = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )

    prompt_ids = tokenizer(
        prompt,
        max_length=MAX_LENGTH,
        padding=False,
        truncation=True,
    )["input_ids"]

    labels = tok["input_ids"].copy()
    prompt_len = len(prompt_ids)

    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100

    tok["labels"] = labels
    return tok


def main():
    set_seed(SEED)

    ds = load_splits_as_datasets()
    tokenizer, model = prepare_tokenizer_and_model()

    tokenized_ds = ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False,
        remove_columns=ds["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        learning_rate=5e-4,
        num_train_epochs=1,        
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
