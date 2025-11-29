# import pandas as pd

# # base = pd.read_csv("../results/samples_base.csv")
# base = pd.read_csv("../results/samples_grpo.csv")
# sft  = pd.read_csv("../results/samples_sft.csv")   # 或 samples_sft.csv，按你的实际路径改

# pred_col = "model_summary"   # 换成你存模型输出那一列的名字

# print("行数：", len(base), len(sft))
# print("预测完全相同吗：", (base[pred_col] == sft[pred_col]).all())
# print("预测相同比例：", (base[pred_col] == sft[pred_col]).mean())

# print("\nBase 第一条：", base[pred_col].iloc[0])
# print("SFT  第一条：", sft[pred_col].iloc[0])



#  extract cleaned summaries


import re
import pandas as pd
from pathlib import Path


def clean_summary(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()

    parts = re.split(r"摘要[:：]\s*", s, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    else:
        return s


def process_file(path_in: str, path_out: str, pred_col: str = "model_summary"):
    print(f"Processing {path_in} -> {path_out}")
    df = pd.read_csv(path_in)

    if pred_col not in df.columns:
        raise ValueError(f"列 '{pred_col}' 不存在，当前列名为: {df.columns.tolist()}")

    df[pred_col] = df[pred_col].astype(str).apply(clean_summary)
    df.to_csv(path_out, index=False)
    print("Done.\n")


if __name__ == "__main__":
    base_dir = Path("../results")

    for name in ["base", "sft", "grpo"]:
        in_path = base_dir / f"samples_{name}.csv"
        out_path = base_dir / f"samples_{name}_clean.csv"
        if in_path.exists():
            process_file(str(in_path), str(out_path))
        else:
            print(f"Skip {in_path}, file not found.")


