import argparse
import os

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BertSentimentDataset, SentimentDataset, load_imdb_with_huggingface, load_local_csv
from models import BERTClassifier, GRUClassifier, LSTMClassifier
from plots import generate_plots
from preprocess import Vocabulary
from utils import compute_metrics, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "gru", "bert"], default="lstm")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--use_hf_imdb", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    # LSTM / GRU args
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=2)
    # BERT args
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len_bert", type=int, default=256)
    # shared
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def evaluate_rnn(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, lengths)
            total_loss += criterion(logits, labels).item()
            all_preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / max(len(loader), 1), 4)
    return metrics


def evaluate_bert(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            total_loss += criterion(logits, labels).item()
            all_preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
            all_labels.extend(labels.long().cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = round(total_loss / max(len(loader), 1), 4)
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_hf_imdb:
        train_df, val_df, test_df = load_imdb_with_huggingface()
    else:
        train_df, val_df, test_df = load_local_csv(args.data_dir)

    best_model_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
    criterion = nn.BCEWithLogitsLoss()

    # BERT
    if args.model == "bert":
        from transformers import BertTokenizer, get_linear_schedule_with_warmup

        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        train_ds = BertSentimentDataset(train_df, tokenizer, max_len=args.max_len_bert)
        val_ds   = BertSentimentDataset(val_df,   tokenizer, max_len=args.max_len_bert)
        test_ds  = BertSentimentDataset(test_df,  tokenizer, max_len=args.max_len_bert)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

        model = BERTClassifier(bert_model_name=args.bert_model, dropout=args.dropout).to(device)

        lr = args.lr if args.lr != 1e-3 else 2e-5  # default BERT lr unless user overrides
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                input_ids     = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels        = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            train_loss = running_loss / max(len(train_loader), 1)
            val_metrics = evaluate_bert(model, val_loader, criterion, device)
            epoch_info = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_score"],
            }
            history.append(epoch_info)
            print(epoch_info)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                tokenizer.save_pretrained(os.path.join(args.save_dir, "bert_tokenizer"))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate_bert(model, test_loader, criterion, device)

    # LSTM / GRU
    else:
        vocab = Vocabulary(max_size=args.vocab_size, min_freq=args.min_freq)
        vocab.build(train_df["text"].tolist())
        vocab_path = os.path.join(args.save_dir, f"vocab_{args.model}.joblib")

        train_ds = SentimentDataset(train_df, vocab, max_len=args.max_len)
        val_ds   = SentimentDataset(val_df,   vocab, max_len=args.max_len)
        test_ds  = SentimentDataset(test_df,  vocab, max_len=args.max_len)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

        cls = LSTMClassifier if args.model == "lstm" else GRUClassifier
        model = cls(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                lengths   = batch["length"].to(device)
                labels    = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(input_ids, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / max(len(train_loader), 1)
            val_metrics = evaluate_rnn(model, val_loader, criterion, device)
            epoch_info = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_score"],
            }
            history.append(epoch_info)
            print(epoch_info)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                joblib.dump(vocab, vocab_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate_rnn(model, test_loader, criterion, device)
        print(f"Saved vocabulary to {vocab_path}")

    print("Test metrics:", test_metrics)
    save_json(history,      os.path.join("results", "metrics", f"history_{args.model}.json"))
    save_json(test_metrics, os.path.join("results", "metrics", f"test_metrics_{args.model}.json"))
    print(f"Saved best model to {best_model_path}")

    generate_plots(args.model)


if __name__ == "__main__":
    main()
