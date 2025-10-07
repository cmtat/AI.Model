"""Tests for modeling dataset assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_model.dataset import build_modeling_dataset


def test_build_modeling_dataset(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    schedule = pd.DataFrame(
        {
            "game_id": ["2023_01_DET_KC", "2023_02_CHI_KC", "2023_03_DEN_KC"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "game_type": ["REG", "REG", "REG"],
            "gameday": ["2023-09-07", "2023-09-14", "2023-09-24"],
            "home_team": ["KC", "KC", "KC"],
            "away_team": ["DET", "CHI", "DEN"],
            "home_score": [20, 31, 28],
            "away_score": [21, 24, 21],
            "result": [-1, 7, 7],
            "spread_line": [-6.5, -8.0, -3.0],
            "home_moneyline": [-260, -300, -180],
            "away_moneyline": [210, 240, 160],
            "div_game": [0, 0, 1],
            "home_rest": [7, 10, 10],
            "away_rest": [7, 9, 7],
            "temp": [75, 70, 60],
            "wind": [5, 3, 10],
        }
    )
    schedule.to_csv(raw_dir / "schedule.csv", index=False)

    elo = pd.DataFrame(
        {
            "date": ["2023-09-07", "2023-09-14", "2023-09-24"],
            "season": [2023, 2023, 2023],
            "team1": ["KC", "KC", "KC"],
            "team2": ["DET", "CHI", "DEN"],
            "elo1_pre": [1660.0, 1650.0, 1670.0],
            "elo2_pre": [1540.0, 1500.0, 1480.0],
            "elo_prob1": [0.70, 0.80, 0.75],
            "qbelo_prob1": [0.68, 0.78, 0.74],
        }
    )
    elo.to_csv(raw_dir / "elo.csv", index=False)

    dataset = build_modeling_dataset(raw_dir)

    assert len(dataset) == 3
    assert dataset.loc[0, "home_team_prev_win"] == 0.0
    assert dataset.loc[1, "home_team_prev_win"] == 0.0
    assert dataset.loc[2, "home_team_prev_win"] == 1.0
    assert dataset.loc[2, "divisional_game"] == 1
    assert dataset["closing_odds"].iloc[0] == -260
    assert dataset["elo_prob_home"].iloc[1] == 0.8
