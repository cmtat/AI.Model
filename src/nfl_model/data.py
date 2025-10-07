"""Data ingestion and basic preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ColumnConfig, PipelineConfig


def load_games(data_path: Path) -> pd.DataFrame:
    """Load historical games from CSV into a DataFrame."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"Data file {data_path} contains no rows.")
    return df


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate that required columns exist."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def get_feature_columns(config: ColumnConfig) -> Tuple[Iterable[str], Iterable[str]]:
    """Return numeric and categorical feature lists."""
    return config.numeric, config.categorical


def split_dataset(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test partitions preserving configuration."""
    target = config.columns.target
    ensure_columns(df, [target])
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=config.split.test_size,
            shuffle=config.split.shuffle,
            random_state=config.split.random_state,
            stratify=df[target],
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=config.split.test_size,
            shuffle=config.split.shuffle,
            random_state=config.split.random_state,
            stratify=None,
        )
    return train_df, test_df


def prepare_inputs(
    df: pd.DataFrame,
    config: PipelineConfig,
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Extract features X and target y from a dataframe."""
    if numeric_features is None or categorical_features is None:
        numeric, categorical = get_feature_columns(config.columns)
    else:
        numeric, categorical = list(numeric_features), list(categorical_features)
    required = list(numeric) + list(categorical)
    if require_target:
        required.append(config.columns.target)
    ensure_columns(df, required)
    X = df[list(numeric) + list(categorical)].copy()
    y = None
    if require_target:
        y = df[config.columns.target].astype(int)
    return X, y


__all__ = [
    "load_games",
    "ensure_columns",
    "get_feature_columns",
    "split_dataset",
    "prepare_inputs",
]
