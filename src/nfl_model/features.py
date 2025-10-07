"""Feature engineering helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import ColumnConfig
from .data import ensure_columns


def american_odds_to_probability(odds: pd.Series) -> pd.Series:
    """Convert American moneyline odds to implied probability."""
    odds = odds.astype(float)
    probabilities = np.where(
        odds < 0,
        -odds / (-odds + 100),
        100 / (odds + 100),
    )
    return pd.Series(probabilities, index=odds.index)


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        median = df[column].median()
        if pd.isna(median):
            median = 0.0
        df[column] = df[column].fillna(median)


def _ensure_categorical(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        df[column] = df[column].fillna("UNKNOWN").astype(str)


def build_features(df: pd.DataFrame, config: ColumnConfig) -> pd.DataFrame:
    """Return dataframe with engineered features appended."""
    numeric = list(config.numeric)
    categorical = list(config.categorical)
    required = set(numeric + categorical)
    ensure_columns(df, required)

    df = df.copy()

    if "home_moneyline" in df.columns and "home_implied_prob" not in df.columns:
        df["home_implied_prob"] = american_odds_to_probability(df["home_moneyline"])
        numeric.append("home_implied_prob")
    if "away_moneyline" in df.columns and "away_implied_prob" not in df.columns:
        df["away_implied_prob"] = american_odds_to_probability(df["away_moneyline"])
        numeric.append("away_implied_prob")

    if "home_spread" in df.columns and "away_spread" in df.columns:
        df["spread_delta"] = df["home_spread"] - (-df["away_spread"])
        numeric.append("spread_delta")

    if "home_implied_prob" in df.columns and "away_implied_prob" in df.columns:
        df["implied_prob_gap"] = df["home_implied_prob"] - df["away_implied_prob"]
        numeric.append("implied_prob_gap")

    _ensure_numeric(df, numeric)
    _ensure_categorical(df, categorical)

    # Deduplicate columns while preserving order.
    df.attrs["numeric_features"] = list(dict.fromkeys(numeric))
    df.attrs["categorical_features"] = list(dict.fromkeys(categorical))
    return df


__all__ = [
    "american_odds_to_probability",
    "build_features",
]
