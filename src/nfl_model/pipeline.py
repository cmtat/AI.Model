"""High-level orchestration for training and inference."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from . import betting
from .config import load_config
from .data import load_games, prepare_inputs, split_dataset
from .features import build_features
from .model import evaluate_model, inference_dataframe, save_model, train_model


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Train model, evaluate on holdout, and persist artifacts."""
    config = load_config(config_path)
    df = load_games(data_path)
    df = build_features(df, config.columns)
    numeric_features = df.attrs.get("numeric_features", config.columns.numeric)
    categorical_features = df.attrs.get("categorical_features", config.columns.categorical)

    train_df, test_df = split_dataset(df, config)
    X_train, y_train = prepare_inputs(
        train_df,
        config,
        numeric_features,
        categorical_features,
    )
    X_test, y_test = prepare_inputs(
        test_df,
        config,
        numeric_features,
        categorical_features,
    )

    model = train_model(
        X_train,
        y_train,
        numeric_features,
        categorical_features,
        config.model,
    )
    metrics = evaluate_model(model, X_test, y_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    evaluation_path = output_dir / "evaluation_predictions.csv"
    config_path_out = output_dir / "config_used.json"

    save_model(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    evaluation_df = test_df.reset_index(drop=True).join(
        inference_dataframe(model, X_test).reset_index(drop=True)
    )
    evaluation_df.to_csv(evaluation_path, index=False)
    with config_path_out.open("w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "evaluation_path": str(evaluation_path),
        "config_path": str(config_path_out),
    }


def generate_predictions(
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    odds_column: Optional[str] = None,
) -> Dict[str, str]:
    """Generate predictions and optional value bet recommendations."""
    from .model import load_model  # Lazy import to avoid circular references.

    config = load_config(config_path)
    if odds_column:
        config.columns.odds = odds_column

    df = load_games(data_path)
    df = build_features(df, config.columns)
    numeric_features = df.attrs.get("numeric_features", config.columns.numeric)
    categorical_features = df.attrs.get("categorical_features", config.columns.categorical)

    X, _ = prepare_inputs(
        df,
        config,
        numeric_features,
        categorical_features,
        require_target=False,
    )
    model = load_model(model_path)
    predictions = inference_dataframe(model, X)

    enriched = df.reset_index(drop=True).join(predictions.reset_index(drop=True))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"predictions_{_timestamp()}.csv"
    enriched.to_csv(predictions_path, index=False)

    artifacts = {
        "predictions_path": str(predictions_path),
    }

    odds_col = odds_column or config.columns.odds
    if odds_col and odds_col in df.columns:
        odds_series = df[odds_col]
        value_bets = betting.calculate_value_bets(
            predictions,
            odds_series,
            config.betting,
        )
        value_path = output_dir / f"value_bets_{_timestamp()}.csv"
        value_bets.to_csv(value_path, index=False)
        artifacts["value_bets_path"] = str(value_path)

    return artifacts


__all__ = [
    "train_and_evaluate",
    "generate_predictions",
]
