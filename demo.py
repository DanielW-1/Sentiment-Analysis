"""
demo.py — Sentiment Analysis Model Comparison Demo
Runs 8 hand-picked test cases through LSTM, GRU, and BERT models and prints
a formatted comparison table with a final accuracy summary.

Usage:
    python3 demo.py
"""

import os
import sys
import re

import torch

# Add src/ to path so we can import project modules directly
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

# ---------------------------------------------------------------------------
# Checkpoint paths (relative to project root)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS = {
    "LSTM": {
        "checkpoint":  os.path.join(BASE_DIR, "checkpoints", "best_lstm.pt"),
        "vocab_path":  os.path.join(BASE_DIR, "checkpoints", "vocab_lstm.joblib"),
    },
    "GRU": {
        "checkpoint":  os.path.join(BASE_DIR, "checkpoints", "best_gru.pt"),
        "vocab_path":  os.path.join(BASE_DIR, "checkpoints", "vocab_gru.joblib"),
    },
    "BERT": {
        "checkpoint":         os.path.join(BASE_DIR, "checkpoints", "best_bert.pt"),
        "tokenizer_dir":      os.path.join(BASE_DIR, "checkpoints", "bert_tokenizer"),
        "bert_model_name":    "bert-base-uncased",
    },
}

# ---------------------------------------------------------------------------
# Demo cases
# ---------------------------------------------------------------------------
DEMO_CASES = [
    {
        "label":        "typical_positive",
        "ground_truth": "positive",
        "text": (
            "This film is an absolute masterpiece. The performances were outstanding, "
            "the story deeply moving, and the direction flawless. "
            "One of the best movies I have ever seen."
        ),
        "note": "Clear positive — all models expected to agree.",
    },
    {
        "label":        "typical_negative",
        "ground_truth": "negative",
        "text": (
            "What a waste of time. The plot made no sense, the acting was wooden, "
            "and the ending left me frustrated. I regret watching this."
        ),
        "note": "Clear negative — all models expected to agree.",
    },
    {
        "label":        "negation",
        "ground_truth": "positive",
        "text": (
            "This movie is not bad at all. I was pleasantly surprised — "
            "I actually quite enjoyed it from start to finish."
        ),
        "note": "Negation test — simple models may misread 'not bad' without context.",
    },
    {
        "label":        "sarcasm",
        "ground_truth": "negative",
        "text": (
            "Oh brilliant, yet another generic superhero sequel nobody asked for. "
            "Totally original. Really loved how nothing new happened for two hours."
        ),
        "note": (
            "Sarcasm test — positive words used ironically; "
            "BERT may catch the negative tone better."
        ),
    },
    {
        "label":        "mixed_sentiment",
        "ground_truth": "negative",
        "text": (
            "The visuals were absolutely stunning and the score was beautiful, "
            "but the story was painfully predictable and the characters felt "
            "completely underdeveloped."
        ),
        "note": (
            "Mixed sentiment — true label is ambiguous; "
            "tests how models weigh competing signals."
        ),
    },
    {
        "label":        "short_positive",
        "ground_truth": "positive",
        "text": "Loved it! Perfect from beginning to end.",
        "note": "Very short input — tests robustness to minimal context.",
    },
    {
        "label":        "subtle_negative",
        "ground_truth": "negative",
        "text": (
            "I wanted to love this film. I really tried. But no matter how hard I looked, "
            "I could not find anything that kept me engaged."
        ),
        "note": "Subtle negative with positive framing — no explicit negative words.",
    },
    {
        "label":        "complex_positive",
        "ground_truth": "positive",
        "text": (
            "What makes this film remarkable is not any single element but the way all "
            "its parts work together. The script, the pacing, the performances — all "
            "cohere into something genuinely special."
        ),
        "note": (
            "Complex positive — long-range coherence needed to identify overall sentiment."
        ),
    },
]

# ---------------------------------------------------------------------------
# Text preprocessing helpers (mirrors src/predict.py)
# ---------------------------------------------------------------------------

def simple_clean(text: str) -> str:
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rnn_model(model_type, cfg, device):
    """Load an LSTM or GRU classifier. Returns (model, vocab) or None on failure."""
    import joblib
    from models import LSTMClassifier, GRUClassifier

    checkpoint = cfg["checkpoint"]
    vocab_path  = cfg["vocab_path"]

    if not os.path.exists(checkpoint):
        print(f"  [WARNING] {model_type} checkpoint not found: {checkpoint} — skipping.")
        return None
    if not os.path.exists(vocab_path):
        print(f"  [WARNING] {model_type} vocab not found: {vocab_path} — skipping.")
        return None

    vocab = joblib.load(vocab_path)
    cls   = LSTMClassifier if model_type == "LSTM" else GRUClassifier
    model = cls(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model, vocab


def load_bert_model(cfg, device):
    """Load the BERT classifier. Returns (model, tokenizer) or None on failure."""
    from transformers import BertTokenizer
    from models import BERTClassifier

    checkpoint    = cfg["checkpoint"]
    tokenizer_dir = cfg["tokenizer_dir"]
    bert_name     = cfg["bert_model_name"]

    if not os.path.exists(checkpoint):
        print(f"  [WARNING] BERT checkpoint not found: {checkpoint} — skipping.")
        return None
    if not os.path.exists(tokenizer_dir):
        print(f"  [WARNING] BERT tokenizer dir not found: {tokenizer_dir} — skipping.")
        return None

    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    model     = BERTClassifier(bert_model_name=bert_name, dropout=0.3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_rnn(text, model, vocab, device, max_len=200):
    """Return (label, confidence_pct) for an RNN model."""
    input_ids, lengths = encode_rnn(text, vocab, max_len=max_len)
    input_ids = input_ids.to(device)
    lengths   = lengths.to(device)
    with torch.no_grad():
        logits = model(input_ids, lengths)
        prob   = torch.sigmoid(logits).item()
    label      = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return label, confidence * 100.0


def predict_bert(text, model, tokenizer, device, max_len=256):
    """Return (label, confidence_pct) for the BERT model."""
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        prob   = torch.sigmoid(logits).item()
    label      = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return label, confidence * 100.0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

WIDTH = 60


def hr(char="="):
    print(char * WIDTH)


def truncate(text, max_chars=70):
    return text if len(text) <= max_chars else text[:max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()

    # ---- Load models -------------------------------------------------------
    print()
    print("Loading models...")

    loaded = {}

    rnn_result = load_rnn_model("LSTM", CHECKPOINTS["LSTM"], device)
    if rnn_result is not None:
        loaded["LSTM"] = rnn_result          # (model, vocab)

    rnn_result = load_rnn_model("GRU", CHECKPOINTS["GRU"], device)
    if rnn_result is not None:
        loaded["GRU"] = rnn_result           # (model, vocab)

    bert_result = load_bert_model(CHECKPOINTS["BERT"], device)
    if bert_result is not None:
        loaded["BERT"] = bert_result         # (model, tokenizer)

    model_names = list(loaded.keys())
    if not model_names:
        print("No models could be loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded: {', '.join(model_names)}  |  Device: {device}\n")

    # ---- Run demo cases ----------------------------------------------------
    # results[case_idx][model_name] = (prediction_label, confidence_pct)
    results = []

    for idx, case in enumerate(DEMO_CASES, start=1):
        text  = case["text"]
        label = case["label"]
        note  = case["note"]

        case_preds = {}
        for name in model_names:
            if name == "BERT":
                model, tokenizer = loaded["BERT"]
                pred, conf = predict_bert(text, model, tokenizer, device)
            else:
                model, vocab = loaded[name]
                pred, conf = predict_rnn(text, model, vocab, device)
            case_preds[name] = (pred, conf)

        results.append(case_preds)

        # Print case block
        hr("=")
        print(f"CASE {idx} — {label}")
        print(f'"{truncate(text)}"')
        hr("-")

        # Pad model names to same width for alignment
        name_w = max(len(n) for n in model_names)
        for name, (pred, conf) in case_preds.items():
            print(f"  {name:<{name_w}}  ->  {pred:<8}  (confidence: {conf:5.1f}%)")

        hr("-")
        print(f"  Note: {note}")

    hr("=")

    # ---- Summary table -----------------------------------------------------
    print()
    print("SUMMARY TABLE")
    hr("=")

    # Header
    name_w = max(len(n) for n in model_names)
    col_w  = max(name_w, 8)       # at least 8 chars wide per model column
    label_w = max(len(c["label"]) for c in DEMO_CASES)

    header_parts = [f"{'#':<3}", f"{'Label':<{label_w}}", f"{'Truth':<8}"]
    for name in model_names:
        header_parts.append(f"{name:^{col_w}}")
    print("  ".join(header_parts))
    hr("-")

    totals_correct = {n: 0 for n in model_names}

    for idx, (case, case_preds) in enumerate(zip(DEMO_CASES, results), start=1):
        gt        = case["ground_truth"].upper()   # "POSITIVE" or "NEGATIVE"
        row_parts = [f"{idx:<3}", f"{case['label']:<{label_w}}", f"{gt:<8}"]

        for name in model_names:
            pred, _conf = case_preds[name]
            correct = pred == gt
            if correct:
                totals_correct[name] += 1
            marker = "OK" if correct else "X"
            row_parts.append(f"{marker:^{col_w}}")

        print("  ".join(row_parts))

    hr("-")

    # Totals row
    total_cases = len(DEMO_CASES)
    totals_parts = [f"{'':3}", f"{'CORRECT':<{label_w}}", f"{'':8}"]
    for name in model_names:
        n = totals_correct[name]
        totals_parts.append(f"{n}/{total_cases}".center(col_w))
    print("  ".join(totals_parts))

    accuracy_parts = [f"{'':3}", f"{'ACCURACY':<{label_w}}", f"{'':8}"]
    for name in model_names:
        acc = totals_correct[name] / total_cases * 100
        accuracy_parts.append(f"{acc:.1f}%".center(col_w))
    print("  ".join(accuracy_parts))

    hr("=")
    print()


if __name__ == "__main__":
    main()
