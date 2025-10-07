"""Value betting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BettingConfig
from .features import american_odds_to_probability


def american_to_decimal(odds: pd.Series) -> pd.Series:
    """Convert American odds to decimal odds."""
    odds = odds.astype(float)
    decimal = np.where(
        odds >= 0,
        1 + (odds / 100.0),
        1 + (100.0 / np.abs(odds)),
    )
    return pd.Series(decimal, index=odds.index)


def kelly_fraction(probability: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    """Compute Kelly fraction for bet sizing."""
    b = decimal_odds - 1.0
    p = probability
    q = 1 - p
    numerator = (b * p) - q
    denominator = b.clip(lower=1e-9)
    frac = numerator / denominator
    frac = frac.clip(lower=0.0)
    return frac


def calculate_value_bets(
    predictions: pd.DataFrame,
    odds: pd.Series,
    config: BettingConfig,
) -> pd.DataFrame:
    """Annotate predictions with value betting indicators."""
    valid = predictions.copy()
    valid = valid[~odds.isna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=list(predictions.columns) + ["implied_probability", "edge", "expected_value", "kelly_fraction", "recommended_stake"])

    implied_prob = american_odds_to_probability(odds.loc[valid.index])
    decimal_odds = american_to_decimal(odds.loc[valid.index])

    probability = valid["home_win_probability"]
    edge = probability - implied_prob
    stake = config.stake_size
    expected_value = (probability * (decimal_odds - 1) - (1 - probability)) * stake
    kelly = kelly_fraction(probability, decimal_odds)

    valid = valid.assign(
        implied_probability=implied_prob,
        edge=edge,
        expected_value=expected_value,
        kelly_fraction=kelly,
        recommended_stake=kelly * stake,
    )
    mask = (edge >= config.min_edge) & (probability >= config.min_probability)
    valid["value_bet"] = mask
    return valid.sort_values(by="edge", ascending=False)


__all__ = [
    "calculate_value_bets",
    "american_to_decimal",
    "kelly_fraction",
]
