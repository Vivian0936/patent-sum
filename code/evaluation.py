"""
evaluation.py

Evaluate a summarization model (Base / SFT / GRPO) on the Chinese-Patent-Summary
dataset using BERTScore. Prints average scores and a few qualitative examples.
"""

import os
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import BERTScorer
from tqdm import tqdm

from data_utils import set_seed, load_splits_as_datasets

# ----------------- config -----------------
#  base, sft, grpo
# MODEL_CHOICE = "grpo"
# MODEL_CHOICE = "base"
MODEL_CHOICE = "sft"

RESULTS_DIR = "../results"

MAX_LENGTH = 2048
SEED = 42
NUM_SAMPLES = 200      # None for full
BATCH_SIZE = 4

PROMPT_TEMPLATE = "请根据以下专利文本生成摘要:\n{text}\n\n摘要:"


def main():
    set_seed(SEED)

    samples_path = os.path.join(RESULTS_DIR, f"samples_{MODEL_CHOICE}_clean.csv")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"找不到 {samples_path}，先跑 slm.py 再评估。")

    df = pd.read_csv(samples_path)

    if NUM_SAMPLES is not None and NUM_SAMPLES < len(df):
        df = df.sample(NUM_SAMPLES, random_state=SEED).reset_index(drop=True)

    refs = df["reference_summary"].tolist()
    hyps = df["model_summary"].tolist()

    print(f"Loaded {len(df)} samples from {samples_path}")
    print("Computing BERTScore...")

    scorer = BERTScorer(lang="zh", rescale_with_baseline=True)
    P, R, F1 = scorer.score(hyps, refs)

    avg_p = P.mean().item()
    avg_r = R.mean().item()
    avg_f1 = F1.mean().item()

    print("\n=== BERTScore Results (from samples CSV) ===")
    print(f"Model:     {MODEL_CHOICE}")
    print(f"Precision: {avg_p:.4f}")
    print(f"Recall:    {avg_r:.4f}")
    print(f"F1:        {avg_f1:.4f}")

    df["bertscore_precision"] = P.cpu().numpy()
    df["bertscore_recall"] = R.cpu().numpy()
    df["bertscore_f1"] = F1.cpu().numpy()

    out_samples = os.path.join(RESULTS_DIR, f"samples_{MODEL_CHOICE}_scored.csv")
    df.to_csv(out_samples, index=False, encoding="utf-8-sig")
    print(f"\nPer-sample scores saved to: {out_samples}")

    summary = {
        "model": MODEL_CHOICE,
        "precision": avg_p,
        "recall": avg_r,
        "f1": avg_f1,
        "num_samples": len(df),
    }
    out_summary = os.path.join(RESULTS_DIR, f"bertscore_{MODEL_CHOICE}.csv")
    pd.DataFrame([summary]).to_csv(out_summary, index=False)
    print(f"BERTScore summary saved to: {out_summary}")


if __name__ == "__main__":
    main()