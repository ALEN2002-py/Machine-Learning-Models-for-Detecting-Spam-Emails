"""
Unit tests for the spam detection pipeline.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import build_models, evaluate_model, load_and_preprocess  # noqa: E402

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Generate a small synthetic dataset that mirrors spambase schema."""
    X, y = make_classification(
        n_samples=500,
        n_features=57,
        n_informative=20,
        n_redundant=10,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(57)])
    df["class"] = y
    # Rename first col to 'word_freq_make' so it loosely resembles spambase
    csv_path = tmp_path / "test_spambase.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ─── Data loading ─────────────────────────────────────────────────────────────


def test_load_and_preprocess_returns_correct_shapes(synthetic_dataset):
    X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess(
        synthetic_dataset
    )
    n_total = 500
    assert len(X_train) + len(X_test) == n_total
    assert X_train.shape[1] == 57
    assert X_test.shape[1] == 57
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(features) == 57


def test_scaler_applied_correctly(synthetic_dataset):
    X_train, X_test, *_ = load_and_preprocess(synthetic_dataset)
    # After StandardScaler, training set mean ≈ 0
    assert abs(X_train.mean()) < 0.1


def test_missing_class_column_raises(tmp_path):
    df = pd.DataFrame(np.random.rand(100, 57))
    path = tmp_path / "no_class.csv"
    df.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="class"):
        load_and_preprocess(str(path))


# ─── Model building ───────────────────────────────────────────────────────────


def test_build_models_returns_three_estimators():
    models = build_models()
    assert len(models) == 3
    assert "Random Forest" in models
    assert "Logistic Regression" in models
    assert "SVM" in models


def test_models_have_fit_predict(synthetic_dataset):
    X_train, X_test, y_train, y_test, *_ = load_and_preprocess(synthetic_dataset)
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test), f"{name}: prediction length mismatch"
        assert set(preds).issubset({0, 1}), f"{name}: unexpected label values"


# ─── Evaluation ───────────────────────────────────────────────────────────────


def test_evaluate_model_returns_required_keys():
    y_test = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.85, 0.15])
    metrics = evaluate_model("Test Model", y_test, y_pred, y_prob)
    for key in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics, f"Missing key: {key}"


def test_evaluate_model_perfect_score():
    y = np.array([0, 1, 0, 1, 1])
    prob = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    metrics = evaluate_model("Perfect", y, y, prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


# ─── Integration ──────────────────────────────────────────────────────────────


def test_end_to_end_random_forest(synthetic_dataset, tmp_path):
    """Quick smoke test: train RF on synthetic data, check metrics JSON is written."""
    from train import build_models, load_and_preprocess

    X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess(
        synthetic_dataset
    )
    models = build_models()
    model = models["Random Forest"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.75, f"Expected accuracy > 0.75 on synthetic data, got {acc:.4f}"
