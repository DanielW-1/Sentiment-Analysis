# Sentiment Analysis of Movie Reviews

This project builds a sentiment analysis system that classifies IMDb movie reviews as **positive** or **negative**. Three models are trained and compared:

* **LSTM** — Long Short-Term Memory with learned word embeddings
* **GRU** — Gated Recurrent Unit with learned word embeddings
* **BERT** — Fine-tuned `bert-base-uncased` transformer

Developed for a Deep Learning for NLP course.

---

## 1. Project Structure

```text
Sentiment-Analysis/
│
├── data/                        # CSV splits (train / val / test)
├── checkpoints/                 # Saved model weights and vocabularies
│   └── bert_tokenizer/          # Saved BERT tokenizer (after training)
├── results/
│   ├── figures/                 # Auto-generated plots (loss, confusion matrix, etc.)
│   └── metrics/                 # JSON files with training history and test metrics
├── src/
│   ├── preprocess.py            # Vocabulary builder for LSTM / GRU
│   ├── dataset.py               # SentimentDataset and BertSentimentDataset
│   ├── models.py                # LSTMClassifier, GRUClassifier, BERTClassifier
│   ├── plots.py                 # Auto-generates figures after every training run
│   ├── utils.py                 # Metrics, seeding, device helpers
│   ├── train.py                 # Training pipeline (all three models)
│   ├── evaluate.py              # Standalone evaluation on the test set
│   └── predict.py               # Single-text inference
├── report/
├── README.md
└── requirements.txt
```

---

## 2. Dataset

This project uses the [IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (25k train / 25k test, balanced).

Two loading options are available:

### Option A: Hugging Face (recommended)

```bash
python src/train.py --model lstm --use_hf_imdb
```

The dataset is downloaded automatically on first run.

### Option B: Local CSV files

Place `train.csv`, `val.csv`, and `test.csv` inside `data/`. Each file must have:

| Column | Description |
|--------|-------------|
| `text` | Review text |
| `label` | `1` = positive, `0` = negative |

```bash
python src/train.py --model lstm --data_dir data
```

---

## 3. Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `torch`, `transformers`, `datasets`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`

---

## 4. Training

Plots are generated automatically at the end of every training run and saved to `results/figures/`. Comparison plots (LSTM vs GRU vs BERT) appear once two or more models have been trained.

### Train LSTM

```bash
python src/train.py --model lstm --use_hf_imdb --epochs 5 --batch_size 64
```

### Train GRU

```bash
python src/train.py --model gru --use_hf_imdb --epochs 5 --batch_size 64
```

### Train BERT

```bash
python src/train.py --model bert --use_hf_imdb --epochs 3 --batch_size 16
```

> BERT uses a lower learning rate (2e-5), AdamW optimizer, linear warmup scheduler, and gradient clipping automatically. Use a smaller batch size due to memory requirements.

After training, each model saves:

* `checkpoints/best_{model}.pt` — best checkpoint by validation loss
* `checkpoints/vocab_{model}.joblib` — vocabulary (LSTM / GRU only)
* `checkpoints/bert_tokenizer/` — tokenizer config (BERT only)
* `results/metrics/history_{model}.json` — per-epoch training history
* `results/metrics/test_metrics_{model}.json` — final test set metrics

---

## 5. Evaluation

Run standalone evaluation on the test set using a saved checkpoint.

### Evaluate LSTM

```bash
python src/evaluate.py \
  --model lstm \
  --checkpoint checkpoints/best_lstm.pt \
  --vocab_path checkpoints/vocab_lstm.joblib \
  --use_hf_imdb
```

### Evaluate GRU

```bash
python src/evaluate.py \
  --model gru \
  --checkpoint checkpoints/best_gru.pt \
  --vocab_path checkpoints/vocab_gru.joblib \
  --use_hf_imdb
```

### Evaluate BERT

```bash
python src/evaluate.py \
  --model bert \
  --checkpoint checkpoints/best_bert.pt \
  --bert_tokenizer_dir checkpoints/bert_tokenizer \
  --use_hf_imdb
```

---

## 6. Prediction / Demo

Run inference on a single custom text.

### LSTM / GRU

```bash
python src/predict.py \
  --model gru \
  --checkpoint checkpoints/best_gru.pt \
  --vocab_path checkpoints/vocab_gru.joblib \
  --text "This movie was absolutely fantastic"
```

### BERT

```bash
python src/predict.py \
  --model bert \
  --checkpoint checkpoints/best_bert.pt \
  --bert_tokenizer_dir checkpoints/bert_tokenizer \
  --text "This movie was absolutely fantastic"
```

Example output:

```text
{'text': 'This movie was absolutely fantastic', 'prediction': 'positive', 'probability_positive': 0.98}
```

---

## 7. Hyperparameters

### LSTM / GRU

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Dropout | 0.3 |
| Batch size | 64 |
| Epochs | 5 |
| Max sequence length | 200 |
| Optimizer | Adam (lr=1e-3) |
| Loss | BCEWithLogitsLoss |

### BERT

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| Dropout | 0.3 |
| Batch size | 16 |
| Epochs | 3 |
| Max sequence length | 256 |
| Optimizer | AdamW (lr=2e-5) |
| Scheduler | Linear warmup (10% of steps) |
| Gradient clipping | 1.0 |
| Loss | BCEWithLogitsLoss |

---

## 8. Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LSTM  | 73.79%   | 73.67%    | 74.05% | 73.86%   |
| GRU   | 82.49%   | 80.17%    | 86.34% | 83.14%   |
| BERT  | TBD after training | | | |

GRU outperforms LSTM by ~9% accuracy despite being a simpler architecture. BERT results will be added after fine-tuning.

---

## 9. Authors

Daniel Wehde & Rami Aabed
