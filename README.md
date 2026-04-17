# Sentiment Analysis of Movie Reviews Using LSTM and GRU

In this project, I built a sentiment analysis system that classifies movie reviews as **positive** or **negative** using two recurrent deep learning models:

* **LSTM**
* **GRU**

This project was developed for a Deep Learning for NLP course. The goal was to keep the implementation simple and clear, while still achieving good performance and allowing easy reproduction of results.

---

## 1. Project Structure

```text
sentiment_project/
│
├── data/
├── results/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── report/
├── README.md
└── requirements.txt
```

---

## 2. Dataset

This project uses the IMDb movie reviews dataset, which contains labeled reviews for sentiment classification.

Two options are available:

### Option A: Hugging Face IMDb dataset

You can directly train using:

```bash
python src/train.py --model lstm --use_hf_imdb
python src/train.py --model gru --use_hf_imdb
```

### Option B: Local CSV files

You can also use locally prepared data by placing these files inside the `data/` folder:

* `train.csv`
* `val.csv`
* `test.csv`

Each file should contain:

* `text` → the review
* `label` → 1 (positive) or 0 (negative)

Example:

```csv
text,label
"This movie was amazing",1
"The film was boring",0
```

Then run:

```bash
python src/train.py --model lstm --data_dir data
python src/train.py --model gru --data_dir data
```

---

## 3. Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 4. Training

### Train LSTM

```bash
python src/train.py --model lstm --use_hf_imdb --epochs 5 --batch_size 64
```

### Train GRU

```bash
python src/train.py --model gru --use_hf_imdb --epochs 5 --batch_size 64
```

After training:

* Models are saved in `checkpoints/`
* Metrics are saved in `results/metrics/`

---

## 5. Evaluation

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

---

## 6. Prediction / Demo

Example for GRU:

```bash
python src/predict.py \
  --model gru \
  --checkpoint checkpoints/best_gru.pt \
  --vocab_path checkpoints/vocab_gru.joblib \
  --text "This movie was amazing"
```

Example output:

```text
{'text': 'This movie was amazing', 'prediction': 'positive', 'probability_positive': 0.97}
```

---

## 7. Suggested Hyperparameters

The following values were used for this project:

* Embedding dimension: `128`
* Hidden dimension: `128`
* Dropout: `0.3`
* Batch size: `64`
* Epochs: `5`
* Max sequence length: `200`
* Optimizer: `Adam`
* Loss function: `BCEWithLogitsLoss`

---

## 8. Results Summary

From the experiments, the GRU model performed better than LSTM.

* LSTM accuracy: ~73.8%
* GRU accuracy: ~82.5%

This shows that even though LSTM is more complex, GRU was able to generalize better for this dataset.

---

## 9. Final Remarks

In this project, I compared LSTM and GRU models for sentiment analysis. From the results, I observed that GRU achieved better performance, while also being faster to train.

This experiment shows that simpler models can sometimes be more effective depending on the task and dataset.

---

## 10. Author

Daniel Wehde & Rami Aabed
