"""Streamlit UI for the NFL modeling workflow."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from nfl_model.dataset import build_modeling_dataset
from nfl_model.ingest import (
    DataIngestionConfig,
    WeatherRequest,
    fetch_data_bundle,
    DataSourceError,
)
from nfl_model import pipeline


st.set_page_config(page_title="NFL Modeling Workbench", layout="wide")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_weather_upload(upload) -> List[WeatherRequest]:
    if upload is None:
        return []
    try:
        df = pd.read_csv(upload)
    except Exception as exc:
        raise ValueError(f"Failed to read weather CSV: {exc}") from exc

    required = {"latitude", "longitude", "start_date", "end_date"}
    if not required.issubset(df.columns):
        raise ValueError(f"Weather CSV must include columns: {sorted(required)}")

    requests: List[WeatherRequest] = []
    for row in df.to_dict(orient="records"):
        requests.append(
            WeatherRequest(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                start_date=str(row["start_date"]),
                end_date=str(row["end_date"]),
                label=str(row.get("label") or f"{row['latitude']}_{row['longitude']}"),
                hourly=tuple(
                    filter(None, (value.strip() for value in str(row.get("hourly", "")).split(",")))
                )
                or ("temperature_2m", "precipitation", "windspeed_10m"),
                daily=tuple(
                    filter(None, (value.strip() for value in str(row.get("daily", "")).split(",")))
                ),
                timezone=str(row.get("timezone", "UTC")),
            )
        )
    return requests


def _write_upload(upload, destination: Path) -> Optional[Path]:
    if upload is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = upload.read()
    destination.write_bytes(data)
    return destination


def _format_summary(summary: Dict[str, int]) -> str:
    buf = StringIO()
    for name, count in summary.items():
        buf.write(f"{name}: {count} rows\n")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def run_fetch(config: DataIngestionConfig) -> Dict[str, int]:
    bundle = fetch_data_bundle(config)
    return {name: len(df) for name, df in bundle.items()}


def run_training(
    raw_dir: Path,
    seasons: Sequence[int],
    artifact_root: Path,
    config_path: Optional[Path],
) -> Dict[str, object]:
    training_df = build_modeling_dataset(raw_dir, seasons=seasons)
    if training_df.empty:
        raise ValueError("Training dataset is empty. Check raw data availability.")

    training_csv = artifact_root / f"training_{seasons[0]}_{seasons[-1]}.csv"
    training_df.to_csv(training_csv, index=False)

    train_artifacts = pipeline.train_and_evaluate(training_csv, artifact_root / "model", config_path)
    metrics_path = Path(train_artifacts["metrics_path"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    evaluation_path = Path(train_artifacts["evaluation_path"])
    evaluation_preview = pd.read_csv(evaluation_path).head(10) if evaluation_path.exists() else pd.DataFrame()

    return {
        "training_csv": training_csv,
        "metrics": metrics,
        "metrics_path": metrics_path,
        "evaluation_path": evaluation_path,
        "evaluation_preview": evaluation_preview,
        "model_path": Path(train_artifacts["model_path"]),
        "config_snapshot": Path(train_artifacts["config_path"]),
    }


def run_predictions(
    raw_dir: Path,
    target_season: int,
    model_path: Path,
    output_dir: Path,
    config_path: Optional[Path],
) -> Dict[str, object]:
    scoring_df = build_modeling_dataset(raw_dir, seasons=[target_season])
    if scoring_df.empty:
        raise ValueError(f"No games found for season {target_season}.")

    scoring_csv = output_dir / f"season_{target_season}_games.csv"
    scoring_df.to_csv(scoring_csv, index=False)
    artifacts = pipeline.generate_predictions(scoring_csv, model_path, output_dir, config_path, odds_column=None)
    predictions_csv = Path(artifacts.get("predictions_path", output_dir / "predictions.csv"))
    predictions_df = pd.read_csv(predictions_csv) if predictions_csv.exists() else pd.DataFrame()
    value_bets_csv = artifacts.get("value_bets_path")
    value_bets_df = pd.read_csv(value_bets_csv) if value_bets_csv and Path(value_bets_csv).exists() else pd.DataFrame()

    return {
        "scoring_csv": scoring_csv,
        "predictions_csv": predictions_csv,
        "predictions_df": predictions_df,
        "value_bets_csv": value_bets_csv,
        "value_bets_df": value_bets_df,
    }


st.title("🏈 NFL Modeling Workbench")
st.write(
    "Interactively fetch data, train the outcome model, and generate per-season predictions. "
    "Configure options in the sidebar and trigger each step from the main panel."
)

with st.sidebar:
    st.header("Configuration")
    train_start = st.number_input("Training start season", min_value=1999, max_value=2100, value=2018, step=1)
    train_end = st.number_input("Training end season", min_value=1999, max_value=2100, value=2024, step=1)
    target_season = st.number_input("Target season to score", min_value=1999, max_value=2100, value=2025, step=1)

    raw_dir_input = st.text_input("Raw data directory", value="data/raw")
    artifact_root_input = st.text_input(
        "Artifact directory",
        value=f"output/season_{target_season}",
        help="Outputs (datasets, models, predictions) will be written here.",
    )

    st.markdown("---")
    st.subheader("Optional data sources")
    odds_api_key = st.text_input("The Odds API key", type="password")
    include_action_network = st.checkbox("Include Action Network public betting data", value=False)
    action_network_user_agent = st.text_input(
        "Action Network user-agent",
        value="nfl-model-streamlit/0.1",
        help="Required if Action Network data is enabled.",
    )

    mysportsfeeds_username = st.text_input("MySportsFeeds username")
    mysportsfeeds_password = st.text_input("MySportsFeeds password/token", type="password")
    weather_upload = st.file_uploader("Weather CSV (optional)", type=["csv"])
    elo_upload = st.file_uploader("Elo snapshot CSV (optional)", type=["csv"])
    config_upload = st.file_uploader("Pipeline YAML config (optional)", type=["yaml", "yml"])

    st.markdown("---")
    st.caption(
        "Set environment variables (e.g., THE_ODDS_API_KEY) if you prefer not to enter keys here. "
        "Inputs are used only for the current session."
    )

if train_end < train_start:
    st.error("Training end season must be >= training start season.")
    st.stop()

raw_dir = Path(raw_dir_input).expanduser()
artifact_root = Path(artifact_root_input).expanduser()
fetch_seasons = list(range(int(train_start), int(max(train_end, target_season)) + 1))

if elo_upload is not None:
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        elo_path = raw_dir / "elo.csv"
        elo_path.write_bytes(elo_upload.getvalue())
        st.session_state["elo_snapshot_path"] = str(elo_path)
        st.sidebar.success(f"Elo snapshot saved to {elo_path}")
    except Exception as exc:
        st.sidebar.error(f"Failed to save Elo snapshot: {exc}")
elif st.session_state.get("elo_snapshot_path"):
    st.sidebar.caption(f"Elo snapshot available: {st.session_state['elo_snapshot_path']}")

st.subheader("1️⃣ Fetch Data")
st.write(
    "Downloads schedules, stats, injuries, and optional feeds for the selected seasons. "
    "Run this step whenever you want to refresh the raw data files."
)

if st.button("Fetch data now", type="primary"):
    try:
        weather_requests = _parse_weather_upload(weather_upload)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    config = DataIngestionConfig(
        seasons=fetch_seasons,
        output_dir=raw_dir,
        odds_api_key=odds_api_key or None,
        mysportsfeeds_season=None,
        mysportsfeeds_username=mysportsfeeds_username or None,
        mysportsfeeds_password=mysportsfeeds_password or None,
        weather_requests=weather_requests,
        action_network_enabled=include_action_network,
        action_network_user_agent=action_network_user_agent or "nfl-model-streamlit/0.1",
    )
    try:
        with st.spinner("Fetching data..."):
            summary = run_fetch(config)
        st.session_state["fetch_summary"] = summary
        st.success("Data fetch complete.")
    except DataSourceError as exc:
        st.error(f"Data fetch failed: {exc}")

if "fetch_summary" in st.session_state:
    st.text(_format_summary(st.session_state["fetch_summary"]))

st.subheader("2️⃣ Train Model")
st.write(
    "Creates the modeling dataset for the selected training span and fits the outcome model. "
    "The resulting metrics and model artifacts are stored in the artifact directory."
)

config_path: Optional[Path] = None
if config_upload is not None:
    config_path = artifact_root / "config_streamlit.yaml"
    _write_upload(config_upload, config_path)

if st.button("Train model", type="primary"):
    try:
        with st.spinner("Training model..."):
            training_result = run_training(
                raw_dir=raw_dir,
                seasons=list(range(int(train_start), int(train_end) + 1)),
                artifact_root=_ensure_dir(artifact_root),
                config_path=config_path,
            )
        st.session_state["training_result"] = training_result
        st.success("Training complete.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")

training_result = st.session_state.get("training_result")
if training_result:
    st.write(f"Training dataset: `{training_result['training_csv']}`")
    if training_result["metrics"]:
        st.json(training_result["metrics"])
    if isinstance(training_result.get("evaluation_preview"), pd.DataFrame) and not training_result[
        "evaluation_preview"
    ].empty:
        st.markdown("**Evaluation sample**")
        st.dataframe(training_result["evaluation_preview"])
    st.write(f"Model artifacts saved to `{training_result['model_path'].parent}`")

st.subheader("3️⃣ Score Target Season")
st.write(
    "Generates predictions for the chosen target season using the newly trained model."
)

if st.button("Run predictions", type="primary"):
    if not training_result:
        st.warning("Train the model first.")
    else:
        try:
            with st.spinner("Generating predictions..."):
                prediction_result = run_predictions(
                    raw_dir=raw_dir,
                    target_season=int(target_season),
                    model_path=training_result["model_path"],
                    output_dir=_ensure_dir(artifact_root / f"predictions_{target_season}"),
                    config_path=config_path,
                )
            st.session_state["prediction_result"] = prediction_result
            st.success("Predictions generated.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

prediction_result = st.session_state.get("prediction_result")
if prediction_result:
    preds_df: pd.DataFrame = prediction_result["predictions_df"]
    if not preds_df.empty:
        st.markdown(f"**Predictions for {target_season}**")
        st.dataframe(preds_df)
        st.download_button(
            label="Download predictions CSV",
            data=preds_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{target_season}.csv",
        )
    value_bets_df: pd.DataFrame = prediction_result["value_bets_df"]
    if not value_bets_df.empty:
        st.markdown("**Value bet opportunities**")
        st.dataframe(value_bets_df)
        st.download_button(
            label="Download value bets CSV",
            data=value_bets_df.to_csv(index=False).encode("utf-8"),
            file_name=f"value_bets_{target_season}.csv",
        )

st.markdown("---")
st.caption(
    "Tip: add `streamlit` to your environment with `pip install -e \".[dev,ingest,ui]\"` "
    "and start the UI via `streamlit run streamlit_app.py`."
)
