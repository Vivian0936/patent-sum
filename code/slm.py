"""
slm.py

Generate summaries from a small language model (e.g., Base, SFT or GRPO model)
on a few random test samples, for qualitative inspection.
"""

import os
import random
import torch
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import set_seed, load_splits_as_datasets


# MODEL_CHOICE = "base"
# MODEL_CHOICE = "sft"
MODEL_CHOICE = "grpo"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SFT_DIR = "../models/sft"
GRPO_DIR = "../models/grpo"
RESULTS_DIR = "../results"
SEED = 42
NUM_EXAMPLES = None
MAX_LENGTH = 2048

PROMPT_TEMPLATE = "请根据以下专利文本生成摘要：\n{text}\n\n摘要:"


def get_model_id():
    if MODEL_CHOICE == "base":
        return BASE_MODEL_NAME          
    elif MODEL_CHOICE == "sft":
        return SFT_DIR                 
    elif MODEL_CHOICE == "grpo":
        return GRPO_DIR                 
    else:
        raise ValueError(f"Unknown MODEL_CHOICE: {MODEL_CHOICE}")

def load_model_and_tokenizer():
    model_id = get_model_id()
    print(f"Loading model from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return tokenizer, model


def generate_single(model, tokenizer, text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    # # greedy method, no sampling
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=256,
    #         num_beams=1,
    #         do_sample=False,
    #         pad_token_id=tokenizer.pad_token_id,
    #     )
    # use sampling and top-k/top-p, temperature
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            # num_beams=1,
            do_sample=True,
            top_p = 0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "摘要：" in full:
        summary = full.split("摘要：", 1)[1].strip()
    else:
        summary = full.replace(prompt, "").strip()
    return summary


def main():
    set_seed(SEED)

    ds = load_splits_as_datasets()
    test_df = ds["test"].to_pandas()

    tokenizer, model = load_model_and_tokenizer()

    if NUM_EXAMPLES is None or NUM_EXAMPLES >= len(test_df):
        sampled = test_df.reset_index(drop=True)
        print(f"Using ALL {len(sampled)} test samples.")
    else:
        sampled = test_df.sample(NUM_EXAMPLES, random_state=SEED).reset_index(drop=True)
        print(f"Sampled {len(sampled)} test samples.")


    rows = []

    for i, row in sampled.iterrows():
        text = row["text"]
        ref = row["summary"]
        gen = generate_single(model, tokenizer, text)

        print("\n" + "=" * 80)
        print(f"[Example {i+1}] ({MODEL_CHOICE})")
        print("\n model summary:", gen)

        rows.append(
            {
                "model": MODEL_CHOICE,
                "text": text,
                "reference_summary": ref,
                "model_summary": gen,
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"samples_{MODEL_CHOICE}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n save to : {out_path}")

if __name__ == "__main__":
    main()
