import argparse

import joblib
import torch

from models import LSTMClassifier, GRUClassifier
from utils import get_device


def simple_clean(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode(text, vocab, max_len=200):
    tokens = simple_clean(text).split()
    ids = [vocab.stoi.get(tok, 1) for tok in tokens[:max_len]]
    length = min(len(tokens), max_len)
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long), torch.tensor([length], dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--text", required=True)
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

    if args.model == "lstm":
        model = LSTMClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
    else:
        model = GRUClassifier(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    input_ids, lengths = encode(args.text, vocab, max_len=args.max_len)
    input_ids = input_ids.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        prob = torch.sigmoid(logits).item()
        label = "positive" if prob >= 0.5 else "negative"

    print({"text": args.text, "prediction": label, "probability_positive": round(prob, 4)})


if __name__ == "__main__":
    main()
