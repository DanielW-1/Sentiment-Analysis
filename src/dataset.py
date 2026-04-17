import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocess import Vocabulary


class SentimentDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vocab: Vocabulary, max_len: int = 200):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, "text"])
        label = int(self.df.loc[idx, "label"])
        input_ids, length = self.vocab.encode(text, max_len=self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


def load_local_csv(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            "Expected train.csv, val.csv, and test.csv inside the data directory."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def load_imdb_with_huggingface():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install the 'datasets' library or provide local CSV files."
        ) from exc

    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    val_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df
