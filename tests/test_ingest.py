"""Tests for data ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_model import data_sources
from nfl_model.ingest import DataIngestionConfig, WeatherRequest, fetch_data_bundle


@pytest.fixture
def stub_sources(monkeypatch):
    """Patch data source functions to avoid network traffic."""

    def make_stub(name: str):
        def _stub(*args, **kwargs):
            return pd.DataFrame({name: [1]})

        return _stub

    monkeypatch.setattr(data_sources, "load_schedule", make_stub("schedule"))
    monkeypatch.setattr(data_sources, "load_weekly_player_stats", make_stub("weekly_player_stats"))
    monkeypatch.setattr(data_sources, "load_injuries", make_stub("injuries"))
    monkeypatch.setattr(data_sources, "load_team_metadata", make_stub("team_metadata"))
    monkeypatch.setattr(data_sources, "load_scoring_lines", make_stub("scoring_lines"))
    monkeypatch.setattr(data_sources, "fetch_fivethirtyeight_elo", make_stub("elo"))
    monkeypatch.setattr(data_sources, "fetch_fivethirtyeight_qb_adjustments", make_stub("qb_elo"))
    monkeypatch.setattr(data_sources, "fetch_the_odds_api", make_stub("odds"))
    monkeypatch.setattr(data_sources, "fetch_mysportsfeeds_injuries", make_stub("mysportsfeeds"))
    monkeypatch.setattr(data_sources, "fetch_action_network_public_bets", make_stub("public_betting"))

    def weather_stub(*args, **kwargs):
        return pd.DataFrame({"time": ["2023-09-10T00:00"], "temperature_2m": [21.0]})

    monkeypatch.setattr(data_sources, "fetch_open_meteo_weather", weather_stub)


def test_fetch_data_bundle_writes_files(tmp_path: Path, stub_sources):
    config = DataIngestionConfig(
        seasons=[2023],
        output_dir=tmp_path,
        odds_api_key="demo",
        mysportsfeeds_season="2024-2025-regular",
        mysportsfeeds_username="user",
        mysportsfeeds_password="pass",
        weather_requests=[
            WeatherRequest(
                latitude=39.7601,
                longitude=-86.1639,
                start_date="2023-09-10",
                end_date="2023-09-10",
                label="indoor-test",
            )
        ],
        action_network_enabled=True,
    )

    bundle = fetch_data_bundle(config)

    expected_keys = {
        "schedule",
        "weekly_player_stats",
        "injuries",
        "team_metadata",
        "scoring_lines",
        "elo",
        "qb_elo",
        "odds",
        "mysportsfeeds_injuries",
        "weather",
        "public_betting",
    }
    assert expected_keys.issubset(bundle.keys())
    # Ensure parquet or csv files written
    output_files = {p.name for p in tmp_path.iterdir()}
    assert any(name.startswith("schedule") for name in output_files)
