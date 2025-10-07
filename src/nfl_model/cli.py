"""Command-line interface for the NFL outcome modeling toolkit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import click

from . import pipeline
from .dataset import build_modeling_dataset
from .ingest import DataIngestionConfig, DataSourceError, WeatherRequest, fetch_data_bundle


def _parse_weather_file(path: Path) -> List[WeatherRequest]:
    """Read weather locations from a CSV file."""
    requests: List[WeatherRequest] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"latitude", "longitude", "start_date", "end_date"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise click.BadParameter(f"Weather CSV must include columns: {sorted(required)}")
        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
                start_date = row["start_date"]
                end_date = row["end_date"]
            except (TypeError, ValueError) as exc:
                raise click.BadParameter("Invalid weather record; check latitude/longitude/start/end dates.") from exc
            label = row.get("label") or f"{latitude}_{longitude}"
            hourly = tuple(
                filter(None, (value.strip() for value in (row.get("hourly") or "").split(",")))
            ) or ("temperature_2m", "precipitation", "windspeed_10m")
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


def _expand_season_range(start: int, end: Optional[int]) -> List[int]:
    """Return list of seasons from start..end inclusive."""
    terminal = end if end is not None else start
    if terminal < start:
        raise click.BadParameter("--end-season must be greater than or equal to --season.")
    return list(range(start, terminal + 1))


def _prepare_game_types(game_types: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(game_types))  # dedupe while preserving order.


@click.group(help="NFL outcome modeling utilities.")
def cli() -> None:
    """CLI entrypoint."""


@cli.command("fetch-data")
@click.option(
    "--season",
    "-s",
    "season_start",
    required=True,
    type=int,
    help="Starting season to ingest (repeat command with --end-season for a range).",
)
@click.option(
    "--end-season",
    "-e",
    type=int,
    default=None,
    show_default=True,
    help="Final season to ingest (inclusive). Defaults to --season.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Directory to store fetched datasets.",
)
@click.option(
    "--odds-api-key",
    envvar="THE_ODDS_API_KEY",
    type=str,
    default=None,
    help="The Odds API key.",
)
@click.option(
    "--mysportsfeeds-season",
    type=str,
    default=None,
    help="Season string for MySportsFeeds (e.g., 2024-2025-regular).",
)
@click.option(
    "--mysportsfeeds-username",
    envvar="MYSPORTSFEEDS_USERNAME",
    type=str,
    default=None,
    help="MySportsFeeds API username.",
)
@click.option(
    "--mysportsfeeds-password",
    envvar="MYSPORTSFEEDS_PASSWORD",
    type=str,
    default=None,
    help="MySportsFeeds API password or token.",
)
@click.option(
    "--weather-locations",
    type=click.Path(path_type=Path),
    default=None,
    help="CSV describing weather queries (latitude,longitude,start_date,end_date,...).",
)
@click.option(
    "--include-action-network",
    is_flag=True,
    default=False,
    help="Fetch Action Network public betting data.",
)
@click.option(
    "--action-network-user-agent",
    type=str,
    default=None,
    help="Custom User-Agent header for Action Network requests.",
)
def fetch_data(
    season_start: int,
    end_season: Optional[int],
    output_dir: Path,
    odds_api_key: Optional[str],
    mysportsfeeds_season: Optional[str],
    mysportsfeeds_username: Optional[str],
    mysportsfeeds_password: Optional[str],
    weather_locations: Optional[Path],
    include_action_network: bool,
    action_network_user_agent: Optional[str],
) -> None:
    """Fetch external datasets required for modeling."""
    seasons = _expand_season_range(season_start, end_season)
    weather_requests: List[WeatherRequest] = []
    if weather_locations:
        weather_requests = _parse_weather_file(weather_locations)

    config = DataIngestionConfig(
        seasons=seasons,
        output_dir=output_dir,
        odds_api_key=odds_api_key,
        mysportsfeeds_season=mysportsfeeds_season,
        mysportsfeeds_username=mysportsfeeds_username,
        mysportsfeeds_password=mysportsfeeds_password,
        weather_requests=weather_requests,
        action_network_enabled=include_action_network,
        action_network_user_agent=action_network_user_agent,
    )

    click.secho("Fetching external datasets...", fg="green")
    try:
        bundle = fetch_data_bundle(config)
    except DataSourceError as exc:
        raise click.ClickException(str(exc)) from exc

    click.secho("Datasets retrieved:", fg="blue")
    for name, frame in bundle.items():
        click.echo(f"  {name}: {len(frame)} rows")


@cli.command("prepare-data")
@click.argument("raw_dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("data/modeling_games.csv"),
    show_default=True,
    help="Where to write the prepared modeling dataset (CSV).",
)
@click.option(
    "--season",
    "-s",
    "seasons",
    multiple=True,
    type=int,
    help="Season(s) to include (repeat flag). Defaults to all seasons present.",
)
@click.option(
    "--game-type",
    "-g",
    "game_types",
    multiple=True,
    type=str,
    default=("REG",),
    show_default=True,
    help="Game types to include (repeat flag, e.g., --game-type REG --game-type POST).",
)
def prepare_data(
    raw_dir: Path,
    output_path: Path,
    seasons: Sequence[int],
    game_types: Sequence[str],
) -> None:
    """Transform raw source tables into a modeling-ready CSV."""
    season_list = list(seasons) or None
    game_type_list = _prepare_game_types(game_types)
    click.secho("Building modeling dataset...", fg="green")
    dataset = build_modeling_dataset(raw_dir, seasons=season_list, game_types=game_type_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    click.secho(f"Dataset written to {output_path}", fg="blue")
    click.echo(f"Rows: {len(dataset)}, Columns: {len(dataset.columns)}")


@cli.command("train")
@click.argument("data_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output"),
    show_default=True,
    help="Directory to write model artifacts.",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional config YAML path.",
)
def train(data_path: Path, output_dir: Path, config: Optional[Path]) -> None:
    """Train and evaluate the model, saving artifacts to disk."""
    click.secho("Starting training run...", fg="green")
    artifacts = pipeline.train_and_evaluate(data_path, output_dir, config)
    click.echo(f"Model saved to: {artifacts['model_path']}")

    metrics_path = Path(artifacts["metrics_path"])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        click.secho("Evaluation metrics:", fg="blue")
        for key, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.4f}")
            else:
                click.echo(f"  {key}: {value}")
    click.echo(f"Evaluation predictions: {artifacts['evaluation_path']}")
    click.echo(f"Config snapshot: {artifacts['config_path']}")


@cli.command("predict")
@click.argument("data_path", type=click.Path(path_type=Path, exists=True))
@click.argument("model_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output/predictions"),
    show_default=True,
    help="Directory for prediction outputs.",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional config YAML path.",
)
@click.option(
    "--odds-column",
    type=str,
    default=None,
    help="Column containing American odds for value-bet calculations.",
)
def predict(
    data_path: Path,
    model_path: Path,
    output_dir: Path,
    config: Optional[Path],
    odds_column: Optional[str],
) -> None:
    """Generate predictions and value bet recommendations."""
    click.secho("Running inference...", fg="green")
    artifacts = pipeline.generate_predictions(
        data_path,
        model_path,
        output_dir,
        config,
        odds_column,
    )
    click.secho("Artifacts generated:", fg="blue")
    for name, path in artifacts.items():
        click.echo(f"  {name}: {path}")


def main() -> None:
    """Entry point for the CLI console script."""
    cli()


if __name__ == "__main__":
    main()
