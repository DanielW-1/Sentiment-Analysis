import argparse

import joblib
import torch

from models import BERTClassifier, GRUClassifier, LSTMClassifier
from utils import get_device


def simple_clean(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_rnn(text, vocab, max_len=200):
    tokens = simple_clean(text).split()
    ids = [vocab.stoi.get(tok, 1) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    lengths = torch.tensor([min(len(tokens), max_len)], dtype=torch.long)
    return torch.tensor([ids], dtype=torch.long), lengths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru", "bert"], default="lstm")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab_path", default=None,
                        help="Required for lstm/gru; not used for bert.")
    parser.add_argument("--bert_tokenizer_dir", default="checkpoints/bert_tokenizer")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--text", required=True)
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

    if args.model == "bert":
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_dir)
        model = BERTClassifier(bert_model_name=args.bert_model, dropout=args.dropout)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device).eval()

        enc = tokenizer(
            args.text,
            max_length=args.max_len_bert,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
            prob = torch.sigmoid(logits).item()

    else:
        if args.vocab_path is None:
            raise ValueError("--vocab_path is required for lstm/gru models.")
        vocab = joblib.load(args.vocab_path)
        cls = LSTMClassifier if args.model == "lstm" else GRUClassifier
        model = cls(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device).eval()

        input_ids, lengths = encode_rnn(args.text, vocab, max_len=args.max_len)
        input_ids = input_ids.to(device)
        lengths   = lengths.to(device)

        with torch.no_grad():
            logits = model(input_ids, lengths)
            prob = torch.sigmoid(logits).item()

    label = "positive" if prob >= 0.5 else "negative"
    print({"text": args.text, "prediction": label, "probability_positive": round(prob, 4)})


if __name__ == "__main__":
    main()
