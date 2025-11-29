"""
gpro.py

Run a small GRPO-style RL fine-tuning on top of the SFT model.

"""

import os
import random
import numpy as np
import torch
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from bert_score import BERTScorer
from collections import Counter
from nltk.util import ngrams
import thulac

from data_utils import set_seed, load_splits_as_datasets


# ------------ config -------------
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
BASE_MODEL_DIR = "../models/sft"     
OUTPUT_DIR = "../models/grpo"
SEED = 42
NUM_RL_SAMPLES = 1000         
PROMPT_TEMPLATE = "请根据以下专利文本生成摘要：\n\n{text}\n\n摘要:"



def build_grpo_dataset(train_df, num_samples=NUM_RL_SAMPLES):
    '''randomly sample some samples from training dataset to be the training data of GRPO'''
    num_samples = min(num_samples, len(train_df))
    indices = random.sample(range(len(train_df)), num_samples)

    dataset = []
    for i in indices:
        original = str(train_df["text"].iloc[i])
        summary = str(train_df["summary"].iloc[i])
        prompt = PROMPT_TEMPLATE.format(text=original)
        dataset.append(
            {
                "prompt": prompt,
                "completion": summary,
                "ground_truth": summary,
            }
        )
    return dataset


# --------------- reward ----------------
thu = thulac.thulac()
device = "cuda" if torch.cuda.is_available() else "cpu"
global_bert_scorer = BERTScorer(
    model_type="bert-base-chinese", 
    lang="zh", 
    rescale_with_baseline=True,
    device = device
    )


def calculate_ngram_diversity(text, n=2):
    tokens = [w[0] for w in thu.cut(text)]
    n_grams = list(ngrams(tokens, n))
    ngram_count = Counter(n_grams)
    return len(ngram_count) / len(n_grams) if n_grams else 0.0


def hybrid_reward(completions, ground_truth, **kwargs):
    """
    completions: list[str] model outputs
    ground_truth: list[str] reference summaries
    """
    rewards = []

    with torch.no_grad():
        P, R, F1 = global_bert_scorer.score(completions, ground_truth, batch_size=8)

    for completion, f1_score in zip(completions, F1):
        content_reward = f1_score.item()
        ngram_div = calculate_ngram_diversity(completion)
        total_reward = 0.7 * content_reward + 0.3 * ngram_div
        rewards.append(total_reward)

    return rewards


def main():
    set_seed(SEED)

    ds = load_splits_as_datasets()
    train_df = ds["train"].to_pandas()
    grpo_dataset = build_grpo_dataset(train_df)

    print(f"GRPO dataset size: {len(grpo_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    config = GRPOConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=3,
        learning_rate=1e-4,
        logging_steps=2,
        max_steps=200,
        output_dir=OUTPUT_DIR,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        reward_funcs=hybrid_reward,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"GRPO model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
