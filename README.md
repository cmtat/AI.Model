# NFL Outcome Modeling Platform

An end-to-end Python 3.12 project for training, evaluating, and operationalizing an NFL game outcome prediction model. The system ingests historical matchup data, builds machine learning features, trains an ensemble classifier, and highlights potential value bets where the model's probabilities diverge from betting odds.

## Features
- Automated training pipeline with configurable data sources and feature engineering.
- Value bet surfacing using implied odds vs. model probabilities.
- CLI interface for backtesting, training, and generating fresh predictions.
- GitHub Actions workflow for scheduled retraining and prediction refreshes.
- Modular architecture ready for extension with new models or betting strategies.

## Project Layout
```
├── data/
│   └── sample_games.csv      # Example dataset structure
├── src/
│   └── nfl_model/
│       ├── cli.py            # Typer CLI entrypoint
│       ├── config.py         # Config dataclasses and defaults
│       ├── data.py           # Data ingestion utilities
│       ├── features.py       # Feature engineering logic
│       ├── model.py          # Model training and evaluation
│       └── pipeline.py       # High-level orchestration
├── tests/
│   ├── test_ingest.py
│   └── test_pipeline.py
├── .github/workflows/
│   └── retrain.yml           # CI automation for retraining
└── pyproject.toml
```

## Quickstart
1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   # Include ingest extras when pulling external feeds:
   # pip install -e ".[dev,ingest]"
   ```
2. **Fetch source data (optional)**
   ```bash
   nfl-model fetch-data --season 2023 --end-season 2024 --output-dir data/raw
   ```
   Add `--odds-api-key` or rely on env vars (`THE_ODDS_API_KEY`, `MYSPORTSFEEDS_USERNAME`, `MYSPORTSFEEDS_PASSWORD`) to enable premium feeds.
3. **Prepare a modeling dataset (optional)**
   ```bash
   nfl-model prepare-data --raw-dir data/raw --output-path data/modeling_games.csv --season 2023 --season 2024
   ```
   The resulting CSV includes derived features (Elo, prior win flags) expected by the model pipeline.
4. **Train and evaluate**
   ```bash
   nfl-model train --data-path data/modeling_games.csv --output-dir output
   ```
   For smoke tests or demos, `data/sample_games.csv` remains available.
5. **Generate predictions and value bets**
   ```bash
   nfl-model predict --data-path data/sample_games.csv --output-dir output --odds-column closing_odds
   ```

Outputs include model artifacts, evaluation metrics, and value bet CSVs under the chosen output directory.

## Data Requirements
- CSV file with historical games, including target column `home_team_win` (1 if home team won, 0 otherwise).
- Example columns demonstrated in `data/sample_games.csv`; extend as needed with additional features.
- Odds columns (moneyline) are optional for training but required for value bet calculations.

## Data Ingestion Sources
The `fetch-data` command stitches together the data sources below. Extend or disable any by passing configuration flags.

| Source | Purpose | Notes |
| --- | --- | --- |
| `nfl_data_py` | Schedules, team metadata, weekly player stats, injuries, betting lines | Install with `pip install .[ingest]` (Python ≤3.11) or pin a compatible release. |
| FiveThirtyEight NFL Elo | Team strength, QB adjustments | No auth required. |
| The Odds API | Real-time moneyline/spread/total markets | Require API key via `--odds-api-key` or `THE_ODDS_API_KEY`. |
| MySportsFeeds | Injury reports, roster updates | Provide `--mysportsfeeds-username` + `--mysportsfeeds-password` and `--mysportsfeeds-season`. |
| Open-Meteo API | Weather snapshots for outdoor stadiums | Supply CSV via `--weather-locations`. |
| Action Network (optional) | Public betting percentages | Enable with `--include-action-network`; set `--action-network-user-agent` if blocked. |

Sample weather CSV (`weather.csv`):
```csv
label,latitude,longitude,start_date,end_date,hourly
gb_lambeau,44.5013,-88.0622,2023-09-10,2023-09-10,temperature_2m,precipitation
```
Then run:
```bash
nfl-model fetch-data --season 2023 --weather-locations weather.csv
nfl-model prepare-data --raw-dir data/raw --output-path data/modeling_games.csv
```

## Preparing Features
`nfl-model prepare-data` merges schedule, betting, and FiveThirtyEight Elo feeds into a modeling-ready table with:
- Binary target `home_team_win` and prior-game win indicators.
- Home/away moneylines and spreads suitable for the default feature configuration.
- Elo strength differentials (`elo_home_pre`, `elo_prob_home`) and optional weather metrics when present.

Use this CSV directly with `nfl-model train` or extend the script to append custom features.

## Full Season Workflow (e.g., 2025)
1. **Fetch raw data covering training + target seasons**
   ```bash
   nfl-model fetch-data --season 2018 --end-season 2025 --output-dir data/raw
   ```
   (Adjust the start year for the amount of history you want.)
2. **Run the season orchestrator**
   ```bash
   nfl-model season-run \
     --train-start 2018 \
     --train-end 2024 \
     --target-season 2025 \
     --raw-dir data/raw \
     --artifact-dir output/season_2025
   ```
   This command writes:
   - `output/season_2025/training_2018_2024.csv` – dataset used for training
   - `output/season_2025/model/` – model artifacts + metrics
   - `output/season_2025/predictions_2025/` – 2025 schedule scored by the trained model

## Streamlit UI
Prefer a graphical interface? Install the UI extras and launch Streamlit:

1. **Install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,ingest,ui]"
   ```
2. **Run**
   ```bash
   streamlit run streamlit_app.py
   ```

The Streamlit workbench mirrors the CLI flow—configure seasons, optional API keys, and output locations in the sidebar, then step through fetching data, training, and generating predictions. Download buttons surface the resulting CSVs directly from the browser.

### Optional YAML Configuration
Override defaults by supplying a YAML file via `--config`:
```yaml
columns:
  target: home_team_win
  odds: home_moneyline
  numeric: ["season", "week", "home_moneyline", "away_moneyline", "home_spread", "away_spread"]
  categorical: ["home_team", "away_team", "divisional_game"]
model:
  estimator: random_forest
  n_estimators: 400
betting:
  min_edge: 0.03
  stake_size: 2.0
```

## Automation
The provided GitHub Actions workflow (`.github/workflows/retrain.yml`) retrains the model on a schedule or manual dispatch, saving artifacts and updating predictions. Customize environment variables or secrets (e.g., API tokens for live data) in the workflow configuration when integrating with production data feeds.
