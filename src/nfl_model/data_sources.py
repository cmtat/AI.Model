"""External data source integrations for NFL modeling."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
import os
import warnings

import pandas as pd
import requests

FIVETHIRTYEIGHT_ELO_URLS = [
    "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv",
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/nfl_elo.csv",
]
FIVETHIRTYEIGHT_QB_ELO_URLS = [
    "https://projects.fivethirtyeight.com/nfl-api/qbelo.csv",
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/nfl-elo/qbelo.csv",
]
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MYSPORTSFEEDS_BASE = "https://api.mysportsfeeds.com/v2.1/pull/nfl"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ACTION_NETWORK_SCOREBOARD_URL = "https://api.actionnetwork.com/web/v1/scoreboard"


class DataSourceError(RuntimeError):
    """Raised when an external data source request fails."""


def _ensure_success(response: requests.Response, context: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise DataSourceError(f"{context} request failed: {exc}") from exc


def _require_nfl_data_py() -> None:
    try:
        import nfl_data_py  # noqa: F401
    except ImportError as exc:
        raise DataSourceError(
            "nfl_data_py is required for this operation. Install with "
            "`pip install nfl-data-py` or `pip install .[ingest]`."
        ) from exc


def load_schedule(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch league schedules for the provided seasons via nfl_data_py."""
    _require_nfl_data_py()
    from nfl_data_py import import_schedules

    return import_schedules(list(seasons))


def load_weekly_player_stats(
    seasons: Sequence[int],
    columns: Optional[Sequence[str]] = None,
    downcast: bool = True,
) -> pd.DataFrame:
    """Fetch weekly player statistics via nfl_data_py."""
    _require_nfl_data_py()
    from nfl_data_py import import_weekly_data

    frames: List[pd.DataFrame] = []
    failures: List[str] = []
    for season in seasons:
        try:
            frames.append(import_weekly_data([season], columns=columns, downcast=downcast))
        except Exception as exc:  # pragma: no cover - depends on remote availability
            failures.append(f"{season}: {exc}")
    if not frames:
        raise DataSourceError(
            f"Weekly data unavailable for requested seasons. Attempts: {failures}"
        )
    if failures:
        warnings.warn(
            "Some seasons were skipped while fetching weekly player stats:\n"
            + "\n".join(failures)
        )

    return pd.concat(frames, ignore_index=True)


def load_injuries(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch league injury reports."""
    _require_nfl_data_py()
    from nfl_data_py import import_injuries

    frames: List[pd.DataFrame] = []
    failures: List[str] = []
    for season in seasons:
        try:
            frames.append(import_injuries([season]))
        except Exception as exc:  # pragma: no cover - remote availability
            failures.append(f"{season}: {exc}")
    if not frames:
        raise DataSourceError(
            f"Injury reports unavailable for requested seasons. Attempts: {failures}"
        )
    if failures:
        warnings.warn(
            "Some seasons were skipped while fetching injury reports:\n" + "\n".join(failures)
        )

    return pd.concat(frames, ignore_index=True)


def load_team_metadata() -> pd.DataFrame:
    """Fetch descriptive team metadata (logos, colors, ids)."""
    _require_nfl_data_py()
    from nfl_data_py import import_team_desc

    return import_team_desc()


def load_scoring_lines(seasons: Sequence[int]) -> pd.DataFrame:
    """Fetch historical scoring lines from nfl_data_py."""
    _require_nfl_data_py()
    from nfl_data_py import import_sc_lines

    frames: List[pd.DataFrame] = []
    failures: List[str] = []
    for season in seasons:
        try:
            frames.append(import_sc_lines([season]))
        except Exception as exc:  # pragma: no cover
            failures.append(f"{season}: {exc}")
    if not frames:
        raise DataSourceError(
            f"Scoring lines unavailable for requested seasons. Attempts: {failures}"
        )
    if failures:
        warnings.warn(
            "Some seasons were skipped while fetching scoring lines:\n" + "\n".join(failures)
        )

    return pd.concat(frames, ignore_index=True)


def _fetch_local_or_env(context: str) -> Optional[pd.DataFrame]:
    """Return dataframe from env-defined snapshot if available."""
    path_override = os.environ.get(f"{context.upper().replace(' ', '_')}_PATH")
    url_override = os.environ.get(f"{context.upper().replace(' ', '_')}_URL")

    if path_override:
        file_path = Path(path_override).expanduser()
        if file_path.exists():
            return pd.read_csv(file_path)
    if url_override:
        try:
            response = requests.get(url_override, timeout=30)
            _ensure_success(response, context)
            return pd.read_csv(StringIO(response.text))
        except Exception:
            warnings.warn(f"Failed to load {context} from override URL.")
    return None


def _fetch_csv_from_urls(urls: Sequence[str], context: str) -> pd.DataFrame:
    errors: List[str] = []
    headers = {"User-Agent": "nfl-model/0.1 (https://github.com/)"}  # polite UA
    for url in urls:
        try:
            response = requests.get(url, timeout=30, headers=headers)
            _ensure_success(response, context)
        except DataSourceError as exc:
            errors.append(f"{url}: {exc}")
            continue

        text = response.text
        if not text or text.lstrip().startswith("<"):
            errors.append(f"{url}: unexpected non-CSV response")
            continue
        try:
            return pd.read_csv(StringIO(text))
        except Exception as exc:  # pragma: no cover - defensive fallback
            errors.append(f"{url}: {exc}")
            continue
    raise DataSourceError(f"{context} download failed. Attempts: {errors}")


def fetch_fivethirtyeight_elo() -> pd.DataFrame:
    """Download FiveThirtyEight Elo ratings."""
    local_df = _fetch_local_or_env("elo_snapshot")
    if local_df is not None:
        return local_df
    return _fetch_csv_from_urls(FIVETHIRTYEIGHT_ELO_URLS, "FiveThirtyEight Elo")


def fetch_fivethirtyeight_qb_adjustments() -> pd.DataFrame:
    """Download FiveThirtyEight QB Elo adjustments."""
    local_df = _fetch_local_or_env("qb_elo_snapshot")
    if local_df is not None:
        return local_df
    return _fetch_csv_from_urls(FIVETHIRTYEIGHT_QB_ELO_URLS, "FiveThirtyEight QB Elo")


def fetch_the_odds_api(
    api_key: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    odds_format: str = "american",
    date_format: str = "iso",
    sport: str = "americanfootball_nfl",
    days_from: Optional[int] = None,
    event_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Pull latest odds snapshots from The Odds API."""
    if not api_key:
        raise ValueError("API key is required for The Odds API.")

    params: Dict[str, str] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if days_from is not None:
        params["daysFrom"] = str(days_from)
    if event_ids:
        params["eventIds"] = ",".join(event_ids)

    url = f"{THE_ODDS_API_BASE}/sports/{sport}/odds"
    response = requests.get(url, params=params, timeout=30)
    _ensure_success(response, "The Odds API")
    payload = response.json()
    return pd.json_normalize(payload)


@dataclass
class MySportsFeedsCredentials:
    username: str
    password: str

    def as_auth_header(self) -> Dict[str, str]:
        token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {token}"}


def fetch_mysportsfeeds(
    season: str,
    endpoint: str,
    credentials: MySportsFeedsCredentials,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Call a MySportsFeeds endpoint and return JSON payload."""
    if not season or not endpoint:
        raise ValueError("Season and endpoint are required for MySportsFeeds requests.")

    url = f"{MYSPORTSFEEDS_BASE}/{season}/{endpoint}.json"
    headers = credentials.as_auth_header()
    response = requests.get(url, headers=headers, params=params or {}, timeout=30)
    _ensure_success(response, "MySportsFeeds")
    return response.json()


def fetch_mysportsfeeds_injuries(
    season: str,
    credentials: MySportsFeedsCredentials,
    params: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch injury reports from MySportsFeeds."""
    payload = fetch_mysportsfeeds(season, "injuries", credentials, params)
    injuries = payload.get("injuries", [])
    return pd.json_normalize(injuries)


def fetch_open_meteo_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly: Sequence[str] = ("temperature_2m", "precipitation", "windspeed_10m"),
    daily: Sequence[str] = (),
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch historical weather data for the given coordinates and date range."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    if hourly:
        params["hourly"] = ",".join(hourly)
    if daily:
        params["daily"] = ",".join(daily)

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    _ensure_success(response, "Open-Meteo")
    data = response.json()
    frames = []
    if "hourly" in data:
        hourly_df = pd.DataFrame(data["hourly"])
        frames.append(hourly_df.assign(resolution="hourly"))
    if "daily" in data:
        daily_df = pd.DataFrame(data["daily"])
        frames.append(daily_df.assign(resolution="daily"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_action_network_public_bets(
    league: str = "nfl",
    user_agent: str = "nfl-model-data-fetcher/0.1",
) -> pd.DataFrame:
    """Fetch public betting percentages from Action Network."""
    url = f"{ACTION_NETWORK_SCOREBOARD_URL}/{league}"
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=30)
    _ensure_success(response, "Action Network")

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise DataSourceError("Action Network returned invalid JSON payload.") from exc

    events = payload.get("games") or payload.get("events") or []
    if not events:
        # Some endpoints wrap data in 'props'
        events = payload.get("props", {}).get("events", [])
    return pd.json_normalize(events)


def write_data_bundle(bundle: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist a dictionary of DataFrames to parquet files."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in bundle.items():
        parquet_path = path / f"{name}.parquet"
        try:
            frame.to_parquet(parquet_path, index=False)
        except (ImportError, ValueError):
            csv_path = path / f"{name}.csv"
            frame.to_csv(csv_path, index=False)


__all__ = [
    "ACTION_NETWORK_SCOREBOARD_URL",
    "DataSourceError",
    "FIVETHIRTYEIGHT_ELO_URL",
    "FIVETHIRTYEIGHT_QB_ELO_URL",
    "MYSPORTSFEEDS_BASE",
    "MySportsFeedsCredentials",
    "OPEN_METEO_ARCHIVE_URL",
    "THE_ODDS_API_BASE",
    "fetch_action_network_public_bets",
    "fetch_fivethirtyeight_elo",
    "fetch_fivethirtyeight_qb_adjustments",
    "fetch_mysportsfeeds",
    "fetch_mysportsfeeds_injuries",
    "fetch_open_meteo_weather",
    "fetch_the_odds_api",
    "load_injuries",
    "load_schedule",
    "load_team_metadata",
    "load_weekly_player_stats",
    "load_scoring_lines",
    "write_data_bundle",
]
