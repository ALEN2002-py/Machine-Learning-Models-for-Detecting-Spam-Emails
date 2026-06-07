"""
Spam Email Detection - Training Pipeline
=========================================
Trains and evaluates four ML models on the UCI Spambase dataset:
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Artificial Neural Network (ANN)

Usage:
    python src/train.py
    python src/train.py --data data/spambase.csv --output results/
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # for cross-validation
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_and_preprocess(data_path: str) -> tuple:
    """
    Load the Spambase dataset and return train/test splits with scaling applied.

    The dataset contains 57 continuous features (word/char frequencies and
    capital-run-length statistics) and a binary target column 'class'
    (1 = spam, 0 = legitimate).

    Parameters
    ----------
    data_path : str
        Path to spambase.csv

    Returns
    -------
    X_train, X_test, y_train, y_test : scaled numpy arrays
    feature_names : list[str]
    scaler : fitted StandardScaler (saved for inference use)
    """
    df = pd.read_csv(data_path)

    # Sanity checks
    assert "class" in df.columns, "Expected target column 'class' not found."
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

    X = df.drop(columns=["class"]).values
    y = df["class"].values
    feature_names = df.drop(columns=["class"]).columns.tolist()

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(
        f"Class distribution — spam: {y.sum()} ({y.mean():.1%}), legit: {(1 - y).sum()} ({(1 - y).mean():.1%})"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Feature scaling (important for LR, SVM, ANN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, scaler


# ─── Models ───────────────────────────────────────────────────────────────────


def build_models() -> dict:
    """Return a dictionary of named, configured sklearn-compatible estimators."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            criterion="entropy",
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,  # enables predict_proba for ROC-AUC
            random_state=RANDOM_STATE,
        ),
    }


def build_ann():
    """Build and compile a Keras ANN model for binary classification."""
    import tensorflow as tf

    tf.random.set_seed(RANDOM_STATE)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(57,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    name: str, y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Compute and return a full metrics dictionary for a single model."""
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:12s}: {v}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Spam'])}"
    )
    return metrics


# ─── Plots ────────────────────────────────────────────────────────────────────


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_confusion_matrix(
    name: str, y_test: np.ndarray, y_pred: np.ndarray, out_dir: Path
):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Legitimate", "Spam"],
        yticklabels=["Legitimate", "Spam"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    _save(fig, out_dir / f"cm_{name.replace(' ', '_').lower()}.png")


def plot_roc_curves(roc_data: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    for i, (name, fpr, tpr, auc) in enumerate(roc_data):
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "roc_curves.png")


def plot_comparison(results: list, out_dir: Path):
    df = pd.DataFrame(results).set_index("model")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[metric], width, label=metric.upper(), color=color)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylim(0.8, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "model_comparison.png")


def plot_feature_importance(
    rf_model, feature_names: list, out_dir: Path, top_n: int = 20
):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="#2196F3",
    )
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Most Discriminative Features (Random Forest)")
    _save(fig, out_dir / "feature_importance.png")


def plot_ann_history(history, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train", color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="#FF5722")
    axes[0].set_title("ANN — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train", color="#2196F3")
    axes[1].plot(history.history["val_loss"], label="Validation", color="#FF5722")
    axes[1].set_title("ANN — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, out_dir / "ann_training_history.png")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(data_path: str = "data/spambase.csv", output_dir: str = "results"):
    import tensorflow as tf

    RESULTS_DIR = Path(output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SPAM EMAIL DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess(
        data_path
    )
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    results = []
    roc_data = []

    # ── Sklearn models ─────────────────────────────────────────────
    models = build_models()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()

        # Cross-validation score
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        print(f"  CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.2f}s")

        metrics = evaluate_model(name, y_test, y_pred, y_prob)
        results.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, metrics["roc_auc"]))

        plot_confusion_matrix(name, y_test, y_pred, RESULTS_DIR)
        joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl")

    # ── ANN ────────────────────────────────────────────────────────
    print("\nTraining ANN...")
    t0 = time.time()

    ann = build_ann()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7
        ),
    ]
    history = ann.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )
    print(
        f"  Training time: {time.time() - t0:.2f}s | Stopped at epoch {len(history.history['loss'])}"
    )

    y_prob_ann = ann.predict(X_test, verbose=0).ravel()
    y_pred_ann = (y_prob_ann >= 0.5).astype(int)

    ann_metrics = evaluate_model("ANN", y_test, y_pred_ann, y_prob_ann)
    results.append(ann_metrics)

    fpr, tpr, _ = roc_curve(y_test, y_prob_ann)
    roc_data.append(("ANN", fpr, tpr, ann_metrics["roc_auc"]))

    plot_confusion_matrix("ANN", y_test, y_pred_ann, RESULTS_DIR)
    plot_ann_history(history, RESULTS_DIR)
    ann.save(MODELS_DIR / "ann.keras")

    # ── Summary plots ──────────────────────────────────────────────
    print("\nGenerating summary plots...")
    plot_roc_curves(roc_data, RESULTS_DIR)
    plot_comparison(results, RESULTS_DIR)
    rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
    plot_feature_importance(rf_model, feature_names, RESULTS_DIR)

    # ── Save metrics JSON ──────────────────────────────────────────
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved → {metrics_path}")

    # ── Final leaderboard ──────────────────────────────────────────
    df_results = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n" + "=" * 60)
    print("  FINAL LEADERBOARD (sorted by F1)")
    print("=" * 60)
    print(df_results.to_string(index=False))

    best = df_results.iloc[0]["model"]
    print(f"\n  Best model: {best} (F1={df_results.iloc[0]['f1']:.4f})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam detection models")
    parser.add_argument(
        "--data", default="data/spambase.csv", help="Path to spambase.csv"
    )
    parser.add_argument(
        "--output", default="results", help="Output directory for results"
    )
    args = parser.parse_args()
    main(args.data, args.output)
