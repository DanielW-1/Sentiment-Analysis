import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_imdb_data(data_dir):
    data = []

    for label in ["pos", "neg"]:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            with open(os.path.join(folder, file), encoding="utf-8") as f:
                text = f.read()
                data.append((text, 1 if label == "pos" else 0))

    return pd.DataFrame(data, columns=["text", "label"])

def main():
    train_path = "aclImdb/train"
    test_path = "aclImdb/test"

    train_df = load_imdb_data(train_path)
    test_df = load_imdb_data(test_path)

    train_df, val_df = train_test_split(train_df, test_size=0.1)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

if __name__ == "__main__":
    main()