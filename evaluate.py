import argparse
import os

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BertSentimentDataset, SentimentDataset, load_imdb_with_huggingface, load_local_csv
from models import BERTClassifier, GRUClassifier, LSTMClassifier
from utils import compute_metrics, get_device, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "gru", "bert"], default="lstm")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Required for lstm/gru; not used for bert.")
    parser.add_argument("--bert_tokenizer_dir", type=str, default="checkpoints/bert_tokenizer")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--use_hf_imdb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--max_len_bert", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    criterion = nn.BCEWithLogitsLoss()

    if args.use_hf_imdb:
        _, _, test_df = load_imdb_with_huggingface()
    else:
        _, _, test_df = load_local_csv(args.data_dir)

    if args.model == "bert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_dir)
        test_ds = BertSentimentDataset(test_df, tokenizer, max_len=args.max_len_bert)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = BERTClassifier(bert_model_name=args.bert_model, dropout=args.dropout)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device).eval()

        total_loss, all_labels, all_preds = 0.0, [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                logits = model(input_ids, attention_mask=attention_mask)
                total_loss += criterion(logits, labels).item()
                all_preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
                all_labels.extend(labels.long().cpu().tolist())

    else:
        if args.vocab_path is None:
            raise ValueError("--vocab_path is required for lstm/gru models.")
        vocab = joblib.load(args.vocab_path)
        test_ds = SentimentDataset(test_df, vocab, max_len=args.max_len)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        cls = LSTMClassifier if args.model == "lstm" else GRUClassifier
        model = cls(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device).eval()

        total_loss, all_labels, all_preds = 0.0, [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                lengths   = batch["length"].to(device)
                labels    = batch["label"].to(device)
                logits = model(input_ids, lengths)
                total_loss += criterion(logits, labels).item()
                all_preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
                all_labels.extend(labels.long().cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / max(len(test_loader), 1), 4)
    print(metrics)
    save_json(metrics, os.path.join("results", "metrics", f"evaluation_{args.model}.json"))


if __name__ == "__main__":
    main()
