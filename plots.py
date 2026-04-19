import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

COLORS = {"lstm": "#4C72B0", "gru": "#DD8452", "bert": "#2CA02C"}
ALL_MODELS = ["lstm", "gru", "bert"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _confusion_heatmap(ax, cm_array, cmap, title):
    cm_pct = cm_array / cm_array.sum(axis=1, keepdims=True)
    annot = np.array(
        [[f"{v}\n({p:.1%})" for v, p in zip(rv, rp)]
         for rv, rp in zip(cm_array, cm_pct)]
    )
    sns.heatmap(
        cm_pct, annot=annot, fmt="", cmap=cmap,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=0.5, cbar=False,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")


def _series(hist, key):
    return [e[key] for e in hist]


def generate_plots(model_name: str, metrics_dir: str = "results/metrics",
                   figures_dir: str = "results/figures"):
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    history = _load_json(metrics_dir / f"history_{model_name}.json")
    test_metrics = _load_json(metrics_dir / f"test_metrics_{model_name}.json")

    epochs = [e["epoch"] for e in history]
    color = COLORS[model_name]
    label = model_name.upper()
    cmap = {"lstm": "Blues", "gru": "Oranges", "bert": "Greens"}[model_name]

    # per-model loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, _series(history, "train_loss"), "o-", color=color, label="Train loss")
    ax.plot(epochs, _series(history, "val_loss"), "s--", color=color, alpha=0.6, label="Val loss")
    ax.set_title(f"{label} — Loss per Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figures_dir / f"loss_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-model val accuracy & F1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in zip(axes, ["val_accuracy", "val_f1"], ["Accuracy", "F1-Score"]):
        ax.plot(epochs, _series(history, key), "o-", color=color)
        ax.set_title(f"{label} — Validation {ylabel} per Epoch", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(epochs)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(figures_dir / f"val_metrics_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-model confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    _confusion_heatmap(ax, np.array(test_metrics["confusion_matrix"]),
                       cmap, f"{label} — Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(figures_dir / f"confusion_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] Saved per-model figures for {label}.")

    # identify which models have complete results
    available = [
        m for m in ALL_MODELS
        if (metrics_dir / f"history_{m}.json").exists()
        and (metrics_dir / f"test_metrics_{m}.json").exists()
    ]

    if len(available) < 2:
        print(f"[plots] Skipping comparison plots (only {label} results available so far).")
        return

    hist_map = {m: _load_json(metrics_dir / f"history_{m}.json") for m in available}
    test_map = {m: _load_json(metrics_dir / f"test_metrics_{m}.json") for m in available}
    title_str = " vs ".join(m.upper() for m in available)

    # comparison: loss curves
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, available):
        h = hist_map[name]
        ep = [e["epoch"] for e in h]
        col = COLORS[name]
        ax.plot(ep, _series(h, "train_loss"), "o-", color=col, label="Train loss")
        ax.plot(ep, _series(h, "val_loss"), "s--", color=col, alpha=0.6, label="Val loss")
        ax.set_title(f"{name.upper()} — Loss per Epoch", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(ep)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Training & Validation Loss", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # comparison: val accuracy & F1 overlaid
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric_key, ylabel, ax in zip(
        ["val_accuracy", "val_f1"], ["Accuracy", "F1-Score"], axes
    ):
        for name in available:
            h = hist_map[name]
            ep = [e["epoch"] for e in h]
            ax.plot(ep, _series(h, metric_key), "o-",
                    color=COLORS[name], label=name.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_title("Validation Accuracy per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_title("Validation F1-Score per Epoch", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "val_accuracy_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # comparison: metrics bar chart
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_keys  = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metric_names))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(available):
        offset = (i - len(available) / 2 + 0.5) * width
        vals = [test_map[name][k] for k in metric_keys]
        bars = ax.bar(x + offset, vals, width, label=name.upper(),
                      color=COLORS[name], edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color=COLORS[name], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Test Set Performance — {title_str}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # overview (2×n grid: loss row + confusion row)
    fig = plt.figure(figsize=(6 * n, 9))
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)

    # top row: individual loss curves
    for col, name in enumerate(available):
        ax = fig.add_subplot(gs[0, col])
        h = hist_map[name]
        ep = [e["epoch"] for e in h]
        col_c = COLORS[name]
        ax.plot(ep, _series(h, "train_loss"), "o-", color=col_c, label="Train")
        ax.plot(ep, _series(h, "val_loss"), "s--", color=col_c, alpha=0.6, label="Val")
        ax.set_title(f"{name.upper()} Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_xticks(ep)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # bottom row: confusion matrices
    cmaps = {"lstm": "Blues", "gru": "Oranges", "bert": "Greens"}
    for col, name in enumerate(available):
        ax = fig.add_subplot(gs[1, col])
        cm = np.array(test_map[name]["confusion_matrix"])
        _confusion_heatmap(ax, cm, cmaps[name], f"{name.upper()} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"Sentiment Analysis — {title_str} Overview",
                 fontsize=15, fontweight="bold")
    fig.savefig(figures_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] Saved comparison figures ({title_str}).")
