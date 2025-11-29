# data_utils.py
import os
import random
import torch
import pandas as pd

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

SEED = 42
DATA_DIR = "../data"   


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_and_save_splits():
    """The initial call: download data from huggingface, and split it, save as csv files"""
    os.makedirs(DATA_DIR, exist_ok=True)

    raw_ds = load_dataset("chenmingxuan/Chinese-Patent-Summary", split="train")
    df = raw_ds.to_pandas()
    df = df.rename(columns={"input": "text", "output": "summary"})

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print("Saved train/val/test to:", DATA_DIR)


def load_splits_as_datasets() -> DatasetDict:

    """For later scripts, we read data from csv file back to DatasetDict"""
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print("CSV splits not found, preparing them now...")
        prepare_and_save_splits()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    ds = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )
    return ds


def main():
    set_seed(SEED)
    prepare_and_save_splits()


if __name__ == "__main__":
    main()