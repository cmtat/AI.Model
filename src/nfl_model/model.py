"""Model creation, training, evaluation, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ModelConfig


def _build_estimator(config: ModelConfig):
    """Instantiate an estimator based on configuration."""
    estimator = config.estimator.lower()
    if estimator == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            class_weight="balanced",
        )
    if estimator == "gradient_boosting":
        return GradientBoostingClassifier(
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
        )
    if estimator == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=config.random_state,
            class_weight="balanced",
        )
    raise ValueError(f"Unsupported estimator: {config.estimator}")


def build_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    model_config: ModelConfig,
) -> Pipeline:
    """Construct preprocessing + estimator pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )
    estimator = _build_estimator(model_config)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    model_config: ModelConfig,
) -> Pipeline:
    """Fit model pipeline."""
    pipeline = build_pipeline(numeric_features, categorical_features, model_config)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float | None]:
    """Return evaluation metrics for the trained model."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    unique_classes = np.unique(y_test)

    results: Dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }
    if len(unique_classes) >= 2:
        results["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        results["log_loss"] = float(log_loss(y_test, probabilities))
    else:
        results["roc_auc"] = None
        results["log_loss"] = None

    results["precision_home"] = float(
        precision_score(y_test, predictions, pos_label=1, zero_division=0)
    )
    results["recall_home"] = float(
        recall_score(y_test, predictions, pos_label=1, zero_division=0)
    )
    results["precision_away"] = float(
        precision_score(y_test, predictions, pos_label=0, zero_division=0)
    )
    results["recall_away"] = float(
        recall_score(y_test, predictions, pos_label=0, zero_division=0)
    )
    return results


def predict_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict home win probabilities."""
    return model.predict_proba(X)[:, 1]


def inference_dataframe(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with predictions and probabilities."""
    probs = predict_probabilities(model, X)
    return pd.DataFrame(
        {
            "home_win_probability": probs,
            "predicted_winner": np.where(probs >= 0.5, "home", "away"),
        },
        index=X.index,
    )


def save_model(model: Pipeline, path: Path) -> None:
    """Persist trained model pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Load a previously saved model."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


__all__ = [
    "train_model",
    "evaluate_model",
    "inference_dataframe",
    "save_model",
    "load_model",
    "build_pipeline",
]
