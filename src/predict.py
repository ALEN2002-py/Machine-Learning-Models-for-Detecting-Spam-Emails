"""
Spam Email Detection — Inference Script
=========================================
Load trained models and run predictions on new feature vectors.

Usage:
    # Predict a single sample (57 comma-separated feature values)
    python src/predict.py --model random_forest --features 0,0.64,0.64,...

    # Batch predict from a CSV (same schema as spambase.csv, without 'class' column)
    python src/predict.py --model random_forest --csv path/to/new_emails.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")

MODEL_REGISTRY = {
    "random_forest": "random_forest.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "svm": "svm.pkl",
    "ann": "ann.keras",
}


def load_model(name: str):
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    path = MODELS_DIR / MODEL_REGISTRY[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run src/train.py first."
        )

    if key == "ann":
        import tensorflow as tf

        return tf.keras.models.load_model(path), "keras"
    return joblib.load(path), "sklearn"


def load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError("Scaler not found. Run src/train.py first.")
    return joblib.load(path)


def predict(model, model_type: str, X: np.ndarray) -> dict:
    if model_type == "keras":
        probs = model.predict(X, verbose=0).ravel()
        labels = (probs >= 0.5).astype(int)
    else:
        probs = model.predict_proba(X)[:, 1]
        labels = model.predict(X)

    return {
        "predictions": labels.tolist(),
        "probabilities": np.round(probs, 4).tolist(),
        "label_names": ["Legitimate" if p == 0 else "Spam" for p in labels],
    }


def main():
    parser = argparse.ArgumentParser(description="Spam email inference")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Which trained model to use",
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Comma-separated 57 feature values for a single email",
    )
    parser.add_argument(
        "--csv", type=str, help="CSV file with feature rows (no 'class' column)"
    )
    args = parser.parse_args()

    if not args.features and not args.csv:
        parser.error("Provide either --features or --csv")

    model, model_type = load_model(args.model)
    scaler = load_scaler()

    if args.features:
        vals = [float(v) for v in args.features.split(",")]
        if len(vals) != 57:
            raise ValueError(f"Expected 57 feature values, got {len(vals)}")
        X = scaler.transform(np.array(vals).reshape(1, -1))
        result = predict(model, model_type, X)
        print(f"\nPrediction : {result['label_names'][0]}")
        print(f"Spam prob  : {result['probabilities'][0]:.4f}\n")

    elif args.csv:
        df = pd.read_csv(args.csv)
        if "class" in df.columns:
            df = df.drop(columns=["class"])
        X = scaler.transform(df.values)
        result = predict(model, model_type, X)
        df["prediction"] = result["label_names"]
        df["spam_probability"] = result["probabilities"]
        out = Path(args.csv).stem + "_predictions.csv"
        df[["prediction", "spam_probability"]].to_csv(out, index=False)
        print(f"\nResults saved → {out}")
        spam_count = sum(1 for p in result["label_names"] if p == "Spam")
        print(
            f"Spam: {spam_count}/{len(result['predictions'])} ({spam_count / len(result['predictions']):.1%})\n"
        )


if __name__ == "__main__":
    main()
