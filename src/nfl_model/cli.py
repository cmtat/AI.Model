"""Typer CLI entrypoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import List, Optional

import typer
from click._utils import UNSET

from . import pipeline
from .dataset import build_modeling_dataset
from .ingest import DataIngestionConfig, DataSourceError, WeatherRequest, fetch_data_bundle

app = typer.Typer(help="NFL outcome modeling utilities.")


def _parse_weather_file(path: Path) -> List[WeatherRequest]:
    requests: List[WeatherRequest] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
                start_date = row["start_date"]
                end_date = row["end_date"]
            except (KeyError, ValueError) as exc:
                raise typer.BadParameter(
                    "Weather file rows must include latitude, longitude, start_date, end_date."
                ) from exc
            label = row.get("label") or f"{latitude}_{longitude}"
            hourly = tuple(filter(None, (value.strip() for value in (row.get("hourly") or "").split(",")))) or (
                "temperature_2m",
                "precipitation",
                "windspeed_10m",
            )
            daily = tuple(filter(None, (value.strip() for value in (row.get("daily") or "").split(","))))
            timezone = row.get("timezone") or "UTC"
            requests.append(
                WeatherRequest(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=start_date,
                    end_date=end_date,
                    label=label,
                    hourly=hourly,
                    daily=daily,
                    timezone=timezone,
                )
            )
    return requests


def _parse_season_values(raw: Optional[str], required: bool = False) -> List[int]:
    if raw is None:
        if required:
            raise typer.BadParameter("Provide at least one --season value.")
        return []
    tokens = [token for token in re.split(r"[\s,]+", raw.strip()) if token]
    if not tokens:
        if required:
            raise typer.BadParameter("Provide at least one --season value.")
        return []
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise typer.BadParameter("Season values must be integers.") from exc


@app.command()
def fetch_data(
    season_start: int = typer.Option(
        ...,
        "--season",
        "-s",
        help="Starting season to ingest (e.g., 2023).",
        flag_value=UNSET,
    ),
    season_end: Optional[int] = typer.Option(
        None,
        "--end-season",
        "-e",
        help="Optional final season (inclusive). Defaults to the starting season.",
        flag_value=UNSET,
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output-dir",
        "-o",
        help="Directory to store fetched datasets.",
        flag_value=UNSET,
    ),
    odds_api_key: Optional[str] = typer.Option(
        None,
        "--odds-api-key",
        envvar="THE_ODDS_API_KEY",
        help="The Odds API key (env THE_ODDS_API_KEY).",
        flag_value=UNSET,
    ),
    mysportsfeeds_season: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-season",
        help="Season string for MySportsFeeds (e.g., 2024-2025-regular).",
        flag_value=UNSET,
    ),
    mysportsfeeds_username: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-username",
        envvar="MYSPORTSFEEDS_USERNAME",
        help="MySportsFeeds API username.",
        flag_value=UNSET,
    ),
    mysportsfeeds_password: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-password",
        envvar="MYSPORTSFEEDS_PASSWORD",
        help="MySportsFeeds API password or token.",
        flag_value=UNSET,
    ),
    weather_locations: Optional[Path] = typer.Option(
        None,
        "--weather-locations",
        help="CSV describing weather queries (latitude,longitude,start_date,end_date,...).",
        flag_value=UNSET,
    ),
    include_action_network: bool = typer.Option(
        False,
        "--include-action-network",
        help="Fetch Action Network public betting data.",
        is_flag=True,
    ),
    action_network_user_agent: Optional[str] = typer.Option(
        None,
        "--action-network-user-agent",
        help="Custom User-Agent header for Action Network requests.",
        flag_value=UNSET,
    ),
) -> None:
    """Fetch and persist external datasets required for modeling."""
    terminal_season = season_end if season_end is not None else season_start
    if terminal_season < season_start:
        raise typer.BadParameter("--end-season must be >= --season.")
    season_values = list(range(season_start, terminal_season + 1))

    weather_requests: List[WeatherRequest] = []
    if weather_locations:
        weather_requests = _parse_weather_file(weather_locations)

    config = DataIngestionConfig(
        seasons=season_values,
        output_dir=output_dir,
        odds_api_key=odds_api_key,
        mysportsfeeds_season=mysportsfeeds_season,
        mysportsfeeds_username=mysportsfeeds_username,
        mysportsfeeds_password=mysportsfeeds_password,
        weather_requests=weather_requests,
        action_network_enabled=include_action_network,
        action_network_user_agent=action_network_user_agent,
    )

    typer.secho("Fetching external datasets...", fg=typer.colors.GREEN)
    try:
        bundle = fetch_data_bundle(config)
    except DataSourceError as exc:
        typer.secho(f"Error fetching data: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho("Datasets retrieved:", fg=typer.colors.BLUE)
    for name, frame in bundle.items():
        typer.echo(f"  {name}: {len(frame)} rows")


@app.command()
def prepare_data(
    raw_dir: Path = typer.Argument(Path("data/raw"), help="Directory of raw data tables."),
    output_path: Path = typer.Option(
        Path("data/modeling_games.csv"),
        "--output-path",
        "-o",
        help="Where to write the prepared modeling dataset (CSV).",
        flag_value=UNSET,
    ),
    season: Optional[str] = typer.Option(
        None,
        "--season",
        "-s",
        help="Season(s) to include (comma or space separated). Defaults to all seasons present.",
        flag_value=UNSET,
    ),
    game_type: List[str] = typer.Option(
        ("REG",),
        "--game-type",
        "-g",
        help="Game types to include (e.g., REG, POST). Repeat to include multiple.",
        is_flag=False,
        flag_value=UNSET,
    ),
) -> None:
    """Transform raw source tables into a modeling-ready CSV."""
    seasons_list = _parse_season_values(season, required=False)
    seasons = seasons_list or None
    game_types = list(game_type) or None
    typer.secho("Building modeling dataset...", fg=typer.colors.GREEN)
    dataset = build_modeling_dataset(raw_dir, seasons=seasons, game_types=game_types)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    typer.secho(f"Dataset written to {output_path}", fg=typer.colors.BLUE)
    typer.echo(f"Rows: {len(dataset)}, Columns: {len(dataset.columns)}")


@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Path to historical training data CSV."),
    output_dir: Path = typer.Option(Path("output"), help="Directory to write model artifacts.", flag_value=UNSET),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config YAML path.", flag_value=UNSET),
) -> None:
    """Train and evaluate the model, saving artifacts to disk."""
    typer.secho("Starting training run...", fg=typer.colors.GREEN)
    artifacts = pipeline.train_and_evaluate(data_path, output_dir, config)
    typer.echo(f"Model saved to: {artifacts['model_path']}")

    metrics_path = Path(artifacts["metrics_path"])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        typer.secho("Evaluation metrics:", fg=typer.colors.BLUE)
        for key, value in metrics.items():
            typer.echo(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    typer.echo(f"Evaluation predictions: {artifacts['evaluation_path']}")
    typer.echo(f"Config snapshot: {artifacts['config_path']}")


@app.command()
def predict(
    data_path: Path = typer.Argument(..., help="CSV containing games to score."),
    model_path: Path = typer.Argument(..., help="Trained model artifact path."),
    output_dir: Path = typer.Option(Path("output/predictions"), help="Directory for prediction outputs.", flag_value=UNSET),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config YAML path.", flag_value=UNSET),
    odds_column: Optional[str] = typer.Option(None, "--odds-column", help="Column containing American odds.", flag_value=UNSET),
) -> None:
    """Generate predictions and value bet recommendations."""
    typer.secho("Running inference...", fg=typer.colors.GREEN)
    artifacts = pipeline.generate_predictions(
        data_path,
        model_path,
        output_dir,
        config,
        odds_column,
    )
    typer.secho("Artifacts generated:", fg=typer.colors.BLUE)
    for name, path in artifacts.items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
