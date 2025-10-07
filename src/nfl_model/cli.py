"""Command-line interface for the NFL outcome modeling toolkit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import click
import pandas as pd

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


@cli.command("season-run")
@click.option("--train-start", type=int, required=True, help="First season to include in the training dataset.")
@click.option("--train-end", type=int, required=True, help="Last season to include in the training dataset.")
@click.option("--target-season", type=int, required=True, help="Season to score for predictions (e.g., 2025).")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    show_default=True,
    help="Directory containing fetched raw datasets.",
)
@click.option(
    "--artifact-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write intermediate artifacts (defaults to output/season_<target>).",
)
@click.option(
    "--config",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional config YAML path for both training and prediction stages.",
)
@click.option(
    "--target-week",
    type=int,
    default=None,
    help="Specific week to score (1-23). If omitted, all games in the target season are scored.",
)
@click.option(
    "--include-target-history/--exclude-target-history",
    default=True,
    show_default=True,
    help="Include games from the target season that have already been played (weeks < target-week) in the training dataset.",
)
def season_run(
    train_start: int,
    train_end: int,
    target_season: int,
    raw_dir: Path,
    artifact_dir: Optional[Path],
    config: Optional[Path],
    target_week: Optional[int],
    include_target_history: bool,
) -> None:
    """Full-season workflow: prepare training data, train, and score a target season."""

    if train_end < train_start:
        raise click.BadParameter("--train-end must be greater than or equal to --train-start.")
    if target_week is not None and not (1 <= target_week <= 23):
        raise click.BadParameter("--target-week must be between 1 and 23.")

    seasons_train = list(range(train_start, train_end + 1))
    artifact_root = artifact_dir or Path(f"output/season_{target_season}")
    artifact_root.mkdir(parents=True, exist_ok=True)

    click.secho(
        f"Preparing training dataset for seasons {train_start}-{train_end}...",
        fg="green",
    )
    training_df = build_modeling_dataset(raw_dir, seasons=seasons_train)
    target_history_df = pd.DataFrame()
    if include_target_history and target_week is not None:
        target_full = build_modeling_dataset(raw_dir, seasons=[target_season])
        target_history_df = target_full[target_full["week"] < target_week]
        if not target_history_df.empty:
            training_df = (
                pd.concat([training_df, target_history_df], ignore_index=True)
                .drop_duplicates(subset="game_id", keep="last")
            )

    if training_df.empty:
        raise click.ClickException(
            "Training dataset is empty. Ensure raw data covers the requested seasons."
    )
    training_csv = artifact_root / f"training_{train_start}_{train_end}.csv"
    training_df.to_csv(training_csv, index=False)

    model_dir = artifact_root / "model"
    click.secho("Training model...", fg="green")
    train_artifacts = pipeline.train_and_evaluate(training_csv, model_dir, config)
    model_path = Path(train_artifacts["model_path"])

    click.secho(f"Building scoring dataset for season {target_season}...", fg="green")
    scoring_df = build_modeling_dataset(raw_dir, seasons=[target_season])
    if target_week is not None:
        scoring_df = scoring_df[scoring_df["week"] == target_week]
    if scoring_df.empty:
        click.secho(
            f"No games found for season {target_season}. Skipping prediction step.", fg="yellow"
        )
        click.echo(f"Training outputs located in {model_dir}")
        return

    scoring_csv = artifact_root / f"season_{target_season}_games.csv"
    scoring_df.to_csv(scoring_csv, index=False)
    predictions_dir = artifact_root / f"predictions_{target_season}"
    prediction_artifacts = pipeline.generate_predictions(
        scoring_csv,
        model_path,
        predictions_dir,
        config,
        odds_column=None,
    )

    click.secho("Season workflow complete.", fg="green")
    click.echo(f"Training dataset: {training_csv}")
    click.echo(f"Model metrics: {train_artifacts['metrics_path']}")
    click.echo(f"Predictions: {prediction_artifacts.get('predictions_path', 'N/A')}")
    if prediction_artifacts.get("value_bets_path"):
        click.echo(f"Value bets: {prediction_artifacts['value_bets_path']}")


def main() -> None:
    """Entry point for the CLI console script."""
    cli()


if __name__ == "__main__":
    main()
