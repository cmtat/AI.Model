from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from nfl_model.cli import cli


def _sample_dataset(season: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"{season}_01_TEAM1_TEAM2"],
            "season": [season],
            "week": [1],
            "gameday": ["2025-09-07"],
            "home_team": ["TEAM1"],
            "away_team": ["TEAM2"],
            "home_score": [0],
            "away_score": [0],
            "home_team_win": [0.0],
            "home_team_prev_win": [0.0],
            "away_team_prev_win": [0.0],
            "home_moneyline": [-110],
            "away_moneyline": [110],
            "home_spread": [-3.0],
            "away_spread": [3.0],
            "divisional_game": [0],
            "home_rest": [7],
            "away_rest": [7],
            "temp": [70],
            "wind": [5],
            "closing_odds": [-110],
            "elo_home_pre": [1550],
            "elo_away_pre": [1500],
            "elo_prob_home": [0.6],
            "qbelo_prob_home": [0.59],
        }
    )


def test_season_run_workflow(monkeypatch, tmp_path):
    training_df = _sample_dataset(2024)
    target_df = _sample_dataset(2025)

    def fake_build(raw_dir, seasons, game_types=None):
        if seasons == [2025]:
            return target_df
        return training_df

    def fake_train(data_path, output_dir, config_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.joblib"
        model_path.write_text("model")
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text("{}")
        evaluation_path = output_dir / "evaluation.csv"
        evaluation_path.write_text("")
        config_used = output_dir / "config.json"
        config_used.write_text("{}")
        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "evaluation_path": str(evaluation_path),
            "config_path": str(config_used),
        }

    def fake_predict(data_path, model_path, output_dir, config_path, odds_column=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "predictions.csv"
        predictions_path.write_text("")
        return {"predictions_path": str(predictions_path)}

    monkeypatch.setattr("nfl_model.cli.build_modeling_dataset", fake_build)
    monkeypatch.setattr("nfl_model.cli.pipeline.train_and_evaluate", fake_train)
    monkeypatch.setattr("nfl_model.cli.pipeline.generate_predictions", fake_predict)

    runner = CliRunner()
    artifacts_dir = tmp_path / "artifacts"
    result = runner.invoke(
        cli,
        [
            "season-run",
            "--train-start",
            "2023",
            "--train-end",
            "2024",
            "--target-season",
            "2025",
            "--raw-dir",
            str(tmp_path),
            "--artifact-dir",
            str(artifacts_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (artifacts_dir / "training_2023_2024.csv").exists()
    assert (artifacts_dir / "model" / "model.joblib").exists()
    assert (artifacts_dir / "predictions_2025" / "predictions.csv").exists()
