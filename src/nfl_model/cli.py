"""Typer CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import pipeline

app = typer.Typer(help="NFL outcome modeling utilities.")


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
