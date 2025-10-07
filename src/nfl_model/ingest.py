"""Ingestion orchestrator for external NFL data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from . import data_sources
from .data_sources import DataSourceError, MySportsFeedsCredentials


@dataclass
class WeatherRequest:
    """Defines a weather query via Open-Meteo."""

    latitude: float
    longitude: float
    start_date: str
    end_date: str
    label: str
    hourly: Sequence[str] = field(
        default_factory=lambda: ("temperature_2m", "precipitation", "windspeed_10m")
    )
    daily: Sequence[str] = field(default_factory=tuple)
    timezone: str = "UTC"


@dataclass
class DataIngestionConfig:
    """Configuration for assembling a modeling dataset."""

    seasons: Sequence[int]
    output_dir: Optional[Path] = None
    odds_api_key: Optional[str] = None
    odds_regions: str = "us"
    odds_markets: str = "h2h,spreads,totals"
    odds_format: str = "american"
    odds_days_from: Optional[int] = None
    mysportsfeeds_season: Optional[str] = None
    mysportsfeeds_username: Optional[str] = None
    mysportsfeeds_password: Optional[str] = None
    weather_requests: Sequence[WeatherRequest] = field(default_factory=list)
    action_network_enabled: bool = False
    action_network_user_agent: Optional[str] = None

    def mysportsfeeds_credentials(self) -> Optional[MySportsFeedsCredentials]:
        if not (self.mysportsfeeds_username and self.mysportsfeeds_password):
            return None
        return MySportsFeedsCredentials(
            username=self.mysportsfeeds_username,
            password=self.mysportsfeeds_password,
        )


def _collect_weather(requests: Sequence[WeatherRequest]) -> pd.DataFrame:
    frames = []
    for req in requests:
        weather = data_sources.fetch_open_meteo_weather(
            latitude=req.latitude,
            longitude=req.longitude,
            start_date=req.start_date,
            end_date=req.end_date,
            hourly=req.hourly,
            daily=req.daily,
            timezone=req.timezone,
        )
        if not weather.empty:
            weather = weather.assign(weather_label=req.label)
        frames.append(weather)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_data_bundle(config: DataIngestionConfig) -> Dict[str, pd.DataFrame]:
    """Fetch all requested datasets based on config."""
    bundle: Dict[str, pd.DataFrame] = {}

    bundle["schedule"] = data_sources.load_schedule(config.seasons)
    bundle["weekly_player_stats"] = data_sources.load_weekly_player_stats(config.seasons)
    bundle["injuries"] = data_sources.load_injuries(config.seasons)
    bundle["team_metadata"] = data_sources.load_team_metadata()
    bundle["scoring_lines"] = data_sources.load_scoring_lines(config.seasons)
    bundle["elo"] = data_sources.fetch_fivethirtyeight_elo()
    bundle["qb_elo"] = data_sources.fetch_fivethirtyeight_qb_adjustments()

    if config.odds_api_key:
        bundle["odds"] = data_sources.fetch_the_odds_api(
            api_key=config.odds_api_key,
            regions=config.odds_regions,
            markets=config.odds_markets,
            odds_format=config.odds_format,
            days_from=config.odds_days_from,
        )
    else:
        bundle["odds"] = pd.DataFrame()

    credentials = config.mysportsfeeds_credentials()
    if config.mysportsfeeds_season and credentials:
        bundle["mysportsfeeds_injuries"] = data_sources.fetch_mysportsfeeds_injuries(
            season=config.mysportsfeeds_season,
            credentials=credentials,
        )
    else:
        bundle["mysportsfeeds_injuries"] = pd.DataFrame()

    if config.weather_requests:
        bundle["weather"] = _collect_weather(config.weather_requests)
    else:
        bundle["weather"] = pd.DataFrame()

    if config.action_network_enabled:
        bundle["public_betting"] = data_sources.fetch_action_network_public_bets(
            user_agent=config.action_network_user_agent or "nfl-model-data-fetcher/0.1",
        )
    else:
        bundle["public_betting"] = pd.DataFrame()

    if config.output_dir:
        data_sources.write_data_bundle(bundle, config.output_dir)

    return bundle


__all__ = [
    "DataIngestionConfig",
    "DataSourceError",
    "WeatherRequest",
    "fetch_data_bundle",
]
