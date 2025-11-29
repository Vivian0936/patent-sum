# annotation.py
'''
This script loads a pre-trained language model to generate summaries for patent texts.
It reads patent data from a CSV file, processes each entry to generate a summary,
and saves the results to a new CSV file.
1. Our main difficulty is AWQ method
2. Within the program, it's more about prompt engineering, we try to design effective prompts
3. and the structured data generation pipeline.
请阅读以下专利说明书，并严格按照下面的 JSON 格式输出，不要出现多余文字：

{
  "background": "...",
  "invention": "...",
  "key_components": ["...", "..."],
  "advantages": "..."
}

专利文本如下：
{TEXT}
and we use pydantic

'''


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import pandas as pd

# 用小一点的指令模型做标注，避免 bitsandbytes/triton 问题
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# 如果你愿意，也可以改成 "Qwen/Qwen2.5-3B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=False,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,   
    device_map={"": device},      
    trust_remote_code=True,
)

model.eval()
print("model loaded on", device)


def generate_summary(text, max_new_tokens=256):
    prompt = f"请根据以下专利文本生成中文摘要\n\n{text}\n\n摘要:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    full = tokenizer.decode(output[0], skip_special_tokens=True)

    if "摘要:" in full:
        summary = full.split("摘要:", 1)[1].strip()
    else:
        summary = full.strip()

    return summary


# ===== read data =====
dataset = pd.read_csv("../data/final_dataset.csv", encoding="utf-8")  # 路径按你真实情况改
print("dataset shape:", dataset.shape)
print("columns:", dataset.columns.tolist())

summaries = []

# To verify the pipeline works, only process the first N entries
N = min(10, len(dataset)) 

for i in range(N):
    story_text = str(dataset["text"][i])
    print(f"\n=== Sample {i+1}/{N} ===")
    print("原文片段:", story_text[:80].replace("\n", " "), "...")
    summary_gen = generate_summary(story_text)
    print("生成摘要:", summary_gen[:120].replace("\n", " "), "...")
    summaries.append(summary_gen)

print(f"\nSummarization done for {N} samples.")

out_path = "../data/summary_debug_small.csv"

with open(out_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    if "text_length" in dataset.columns:
        writer.writerow(["text", "summary", "text_length"])
        for i in range(N):
            writer.writerow([dataset["text"][i], summaries[i], dataset["text_length"][i]])
    else:
        writer.writerow(["text", "summary"])
        for i in range(N):
            writer.writerow([dataset["text"][i], summaries[i]])

print(f"CSV file saved as {out_path}")
