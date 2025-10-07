"""Basic smoke tests for the training and betting workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nfl_model import betting, pipeline
from nfl_model.config import BettingConfig

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_games.csv"


def test_train_and_evaluate(tmp_path):
    artifacts = pipeline.train_and_evaluate(DATA_PATH, tmp_path)
    model_path = Path(artifacts["model_path"])
    metrics_path = Path(artifacts["metrics_path"])
    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_value_bet_calculation():
    predictions = pd.DataFrame(
        {
            "home_win_probability": [0.65, 0.45],
        }
    )
    odds = pd.Series([-110, 130])
    config = BettingConfig(min_edge=0.01, min_probability=0.4, stake_size=1.0)
    bets = betting.calculate_value_bets(predictions, odds, config)

    assert "value_bet" in bets.columns
    assert bets.iloc[0]["value_bet"]
    assert bets["edge"].iloc[0] >= 0.01
