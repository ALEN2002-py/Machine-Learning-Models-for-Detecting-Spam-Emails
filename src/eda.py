"""
Spam Email Detection — Exploratory Data Analysis
=================================================
Produces distribution plots, correlation heatmaps, and class-balance
visualisations for the UCI Spambase dataset.

Usage:
    python src/eda.py
    python src/eda.py --data data/spambase.csv --output results/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def run_eda(data_path: str = "data/spambase.csv", output_dir: str = "results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"\nDataset shape : {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(
        f"\nClass distribution:\n{df['class'].value_counts().rename({0: 'Legitimate', 1: 'Spam'})}"
    )

    # ── Class balance ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["class"].value_counts().sort_index()
    ax.bar(["Legitimate", "Spam"], counts.values, color=["#4CAF50", "#F44336"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    save(fig, out / "eda_class_distribution.png")

    # ── Feature-type correlation with label ──────────────────────────
    corr = df.corr()["class"].drop("class").abs().sort_values(ascending=False)
    top20 = corr.head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    top20.plot(kind="barh", ax=ax, color="#2196F3")
    ax.invert_yaxis()
    ax.set_xlabel("Absolute Pearson Correlation with Spam Label")
    ax.set_title("Top 20 Features Correlated with Spam")
    save(fig, out / "eda_feature_correlations.png")

    # ── Feature group distributions ──────────────────────────────────
    word_cols = [c for c in df.columns if c.startswith("word_freq")][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()
    for i, col in enumerate(word_cols):
        spam = df[df["class"] == 1][col]
        legit = df[df["class"] == 0][col]
        axes[i].hist(
            legit, bins=40, alpha=0.6, label="Legitimate", color="#4CAF50", density=True
        )
        axes[i].hist(
            spam, bins=40, alpha=0.6, label="Spam", color="#F44336", density=True
        )
        axes[i].set_title(col.replace("word_freq_", ""), fontsize=8)
        axes[i].set_xlabel("Frequency")
        if i == 0:
            axes[i].legend(fontsize=7)
    fig.suptitle("Word Frequency Distributions — Spam vs Legitimate", fontsize=12)
    fig.tight_layout()
    save(fig, out / "eda_word_distributions.png")

    # ── Correlation heatmap (char + capital features) ────────────────
    special_cols = [
        c for c in df.columns if c.startswith("char_freq") or c.startswith("capital")
    ]
    special_cols.append("class")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[special_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix — Character & Capital Features")
    save(fig, out / "eda_char_capital_correlations.png")

    print("\nEDA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spambase.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_eda(args.data, args.output)
