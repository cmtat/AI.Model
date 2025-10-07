"""Typer CLI entrypoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

import typer

from . import pipeline
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


@app.command()
def fetch_data(
    season: List[int] = typer.Option(
        ...,
        "--season",
        "-s",
        help="Season(s) to ingest (repeat for multiple years).",
    ),
    output_dir: Path = typer.Option(
        Path("data/raw"),
        "--output-dir",
        "-o",
        help="Directory to store fetched datasets.",
    ),
    odds_api_key: Optional[str] = typer.Option(
        None,
        "--odds-api-key",
        envvar="THE_ODDS_API_KEY",
        help="The Odds API key (env THE_ODDS_API_KEY).",
    ),
    mysportsfeeds_season: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-season",
        help="Season string for MySportsFeeds (e.g., 2024-2025-regular).",
    ),
    mysportsfeeds_username: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-username",
        envvar="MYSPORTSFEEDS_USERNAME",
        help="MySportsFeeds API username.",
    ),
    mysportsfeeds_password: Optional[str] = typer.Option(
        None,
        "--mysportsfeeds-password",
        envvar="MYSPORTSFEEDS_PASSWORD",
        help="MySportsFeeds API password or token.",
    ),
    weather_locations: Optional[Path] = typer.Option(
        None,
        "--weather-locations",
        help="CSV describing weather queries (latitude,longitude,start_date,end_date,...).",
    ),
    include_action_network: bool = typer.Option(
        False,
        "--include-action-network/--skip-action-network",
        help="Fetch Action Network public betting data.",
    ),
    action_network_user_agent: Optional[str] = typer.Option(
        None,
        "--action-network-user-agent",
        help="Custom User-Agent header for Action Network requests.",
    ),
) -> None:
    """Fetch and persist external datasets required for modeling."""
    weather_requests: List[WeatherRequest] = []
    if weather_locations:
        weather_requests = _parse_weather_file(weather_locations)

    config = DataIngestionConfig(
        seasons=season,
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
def train(
    data_path: Path = typer.Argument(..., help="Path to historical training data CSV."),
    output_dir: Path = typer.Option(Path("output"), help="Directory to write model artifacts."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config YAML path."),
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
    output_dir: Path = typer.Option(Path("output/predictions"), help="Directory for prediction outputs."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional config YAML path."),
    odds_column: Optional[str] = typer.Option(None, "--odds-column", help="Column containing American odds."),
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
