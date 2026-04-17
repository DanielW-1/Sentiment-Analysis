import argparse
import os

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SentimentDataset, load_imdb_with_huggingface, load_local_csv
from models import LSTMClassifier, GRUClassifier
from utils import compute_metrics, get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--use_hf_imdb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    vocab = joblib.load(args.vocab_path)

    if args.use_hf_imdb:
        _, _, test_df = load_imdb_with_huggingface()
    else:
        _, _, test_df = load_local_csv(args.data_dir)

    test_ds = SentimentDataset(test_df, vocab, max_len=args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "lstm":
        model = LSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
    else:
        model = GRUClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / max(len(test_loader), 1), 4)
    print(metrics)
    save_json(metrics, os.path.join("results", "metrics", f"evaluation_{args.model}.json"))


if __name__ == "__main__":
    main()
