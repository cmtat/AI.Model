"""Utilities to assemble modeling datasets from raw league sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

TEAM_ABBREV_TO_ELO = {
    "JAX": "JAC",
    "WSH": "WAS",
    "LAR": "LA",
    "STL": "STL",
    "SD": "SD",
    "OAK": "OAK",
    "LA": "LA",
    "LV": "LV",
}


def _load_table(raw_dir: Path, name: str) -> pd.DataFrame:
    """Load a table from raw data directory supporting csv/json/parquet."""
    candidates = [
        raw_dir / f"{name}.parquet",
        raw_dir / f"{name}.csv",
        raw_dir / f"{name}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except (ImportError, ValueError) as exc:
                raise RuntimeError(
                    f"Unable to read parquet file {path}. Install 'pyarrow' or 'fastparquet'."
                ) from exc
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            return pd.read_json(path)
    return pd.DataFrame()


def _normalize_team(team: str) -> str:
    if not isinstance(team, str):
        return team
    team = team.upper()
    return TEAM_ABBREV_TO_ELO.get(team, team)


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _compute_home_win(row: pd.Series) -> float:
    if "result" in row and not pd.isna(row["result"]):
        return float(row["result"] > 0)
    if "home_score" in row and "away_score" in row:
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return np.nan
        if row["home_score"] == row["away_score"]:
            return 0.0
        return float(row["home_score"] > row["away_score"])
    return np.nan


def _prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"game_id", "season", "week", "home_team", "away_team"}
    missing = required_columns - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule dataset missing required columns: {missing}")

    schedule = schedule.copy()
    date_column = "gameday" if "gameday" in schedule.columns else "game_date"
    if date_column not in schedule:
        raise ValueError("Schedule dataset must include 'gameday' or 'game_date'.")
    schedule["gameday"] = pd.to_datetime(schedule[date_column])

    if "game_type" in schedule.columns:
        schedule["game_type"] = schedule["game_type"].fillna("REG")

    schedule["home_team_win"] = schedule.apply(_compute_home_win, axis=1).fillna(0.0)
    if "result" in schedule.columns:
        schedule["away_team_win"] = (schedule["result"] < 0).astype(float)
    elif {"home_score", "away_score"} <= set(schedule.columns):
        schedule["away_team_win"] = (schedule["away_score"] > schedule["home_score"]).astype(float)
    else:
        schedule["away_team_win"] = 1.0 - schedule["home_team_win"]

    _ensure_numeric(
        schedule,
        [
            "spread_line",
            "total_line",
            "home_moneyline",
            "away_moneyline",
            "temp",
            "wind",
            "home_rest",
            "away_rest",
        ],
    )

    schedule["home_spread"] = -schedule.get("spread_line", 0)
    schedule["away_spread"] = schedule.get("spread_line", 0)

    if "div_game" in schedule.columns:
        schedule["divisional_game"] = schedule["div_game"].astype(int)
    elif "divisional_game" not in schedule.columns:
        schedule["divisional_game"] = 0

    schedule["closing_odds"] = schedule.get("home_moneyline")

    return schedule


def _attach_previous_outcomes(schedule: pd.DataFrame) -> pd.DataFrame:
    long_frames: List[pd.DataFrame] = []

    home = schedule[["game_id", "season", "week", "home_team", "home_team_win"]].copy()
    home.rename(
        columns={
            "home_team": "team",
            "home_team_win": "win",
        },
        inplace=True,
    )
    home["side"] = "home"

    away = schedule[["game_id", "season", "week", "away_team", "away_team_win"]].copy()
    away.rename(
        columns={
            "away_team": "team",
            "away_team_win": "win",
        },
        inplace=True,
    )
    away["side"] = "away"

    long_frames.extend([home, away])
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df.sort_values(["team", "season", "week"], inplace=True)
    long_df["prev_win"] = long_df.groupby("team")["win"].shift(1).fillna(0.0)

    home_prev = long_df[long_df["side"] == "home"][["game_id", "prev_win"]].rename(
        columns={"prev_win": "home_team_prev_win"}
    )
    away_prev = long_df[long_df["side"] == "away"][["game_id", "prev_win"]].rename(
        columns={"prev_win": "away_team_prev_win"}
    )

    schedule = schedule.merge(home_prev, on="game_id", how="left")
    schedule = schedule.merge(away_prev, on="game_id", how="left")
    schedule["home_team_prev_win"] = schedule["home_team_prev_win"].fillna(0.0)
    schedule["away_team_prev_win"] = schedule["away_team_prev_win"].fillna(0.0)
    return schedule


def _attach_elo(schedule: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    elo = _load_table(raw_dir, "elo")
    if elo.empty:
        schedule["elo_home_pre"] = np.nan
        schedule["elo_away_pre"] = np.nan
        schedule["elo_prob_home"] = np.nan
        schedule["qbelo_prob_home"] = np.nan
        return schedule

    elo = elo.copy()
    if "date" not in elo.columns:
        raise ValueError("FiveThirtyEight Elo dataset missing 'date' column.")
    elo["date"] = pd.to_datetime(elo["date"])
    required = {"season", "team1", "team2"}
    missing = required - set(elo.columns)
    if missing:
        raise ValueError(f"Elo dataset missing required columns: {missing}")

    schedule = schedule.copy()
    schedule["home_team_elo"] = schedule["home_team"].map(_normalize_team)
    schedule["away_team_elo"] = schedule["away_team"].map(_normalize_team)

    merge_keys = ["season", "gameday", "home_team_elo", "away_team_elo"]
    elo_primary = elo.rename(
        columns={
            "team1": "home_team_elo",
            "team2": "away_team_elo",
            "elo1_pre": "elo_home_pre",
            "elo2_pre": "elo_away_pre",
            "elo_prob1": "elo_prob_home",
            "qbelo_prob1": "qbelo_prob_home",
            "date": "gameday",
        }
    )

    merged = schedule.merge(
        elo_primary,
        on=["season", "gameday", "home_team_elo", "away_team_elo"],
        how="left",
        suffixes=("", "_elo"),
    )

    missing_mask = merged["elo_home_pre"].isna()
    if missing_mask.any():
        # Attempt swapped merge where elo lists teams in the opposite order.
        elo_swapped = elo.rename(
            columns={
                "team1": "away_team_elo",
                "team2": "home_team_elo",
                "elo1_pre": "elo_away_pre",
                "elo2_pre": "elo_home_pre",
                "elo_prob1": "elo_prob_home_swapped",
                "qbelo_prob1": "qbelo_prob_home_swapped",
                "date": "gameday",
            }
        )
        alt = schedule.loc[missing_mask].merge(
            elo_swapped,
            on=["season", "gameday", "home_team_elo", "away_team_elo"],
            how="left",
            suffixes=("", "_elo"),
        )
        if not alt.empty:
            merged.loc[missing_mask, "elo_home_pre"] = alt["elo_home_pre"].values
            merged.loc[missing_mask, "elo_away_pre"] = alt["elo_away_pre"].values
            merged.loc[missing_mask, "elo_prob_home"] = 1 - alt["elo_prob_home_swapped"].values
            if "qbelo_prob_home_swapped" in alt:
                merged.loc[missing_mask, "qbelo_prob_home"] = 1 - alt["qbelo_prob_home_swapped"].values

    for col in ["elo_home_pre", "elo_away_pre", "elo_prob_home", "qbelo_prob_home"]:
        if col not in merged:
            merged[col] = np.nan

    merged.drop(
        columns=[
            c
            for c in [
                "home_team_elo",
                "away_team_elo",
                "qbelo_prob_home_swapped",
                "elo_prob_home_swapped",
            ]
            if c in merged.columns
        ],
        inplace=True,
    )

    return merged


def build_modeling_dataset(
    raw_dir: Path,
    seasons: Optional[Sequence[int]] = None,
    game_types: Optional[Sequence[str]] = ("REG",),
) -> pd.DataFrame:
    """Create a modeling dataset from previously fetched raw data."""

    raw_dir = Path(raw_dir)
    schedule = _load_table(raw_dir, "schedule")
    if schedule.empty:
        raise FileNotFoundError(f"No schedule dataset found in {raw_dir}")

    schedule = _prepare_schedule(schedule)

    if seasons:
        schedule = schedule[schedule["season"].isin(seasons)]
    if game_types:
        schedule = schedule[schedule["game_type"].isin(game_types)]

    schedule = _attach_previous_outcomes(schedule)
    schedule = _attach_elo(schedule, raw_dir)

    columns_to_keep = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_team_win",
        "home_team_prev_win",
        "away_team_prev_win",
        "home_moneyline",
        "away_moneyline",
        "home_spread",
        "away_spread",
        "divisional_game",
        "home_rest",
        "away_rest",
        "temp",
        "wind",
        "closing_odds",
        "elo_home_pre",
        "elo_away_pre",
        "elo_prob_home",
        "qbelo_prob_home",
    ]
    for column in columns_to_keep:
        if column not in schedule.columns:
            schedule[column] = np.nan

    dataset = schedule[columns_to_keep].copy()
    dataset.sort_values(["season", "week", "game_id"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def write_modeling_dataset(
    raw_dir: Path,
    output_path: Path,
    seasons: Optional[Sequence[int]] = None,
    game_types: Optional[Sequence[str]] = ("REG",),
) -> Path:
    """Build and persist the modeling dataset to disk."""
    dataset = build_modeling_dataset(raw_dir, seasons=seasons, game_types=game_types)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


__all__ = [
    "build_modeling_dataset",
    "write_modeling_dataset",
]
