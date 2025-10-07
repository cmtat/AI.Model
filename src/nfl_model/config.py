"""Configuration objects and utilities for the NFL outcome pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ColumnConfig:
    """Column assignments for dataset ingestion and feature building."""

    target: str = "home_team_win"
    game_id: Optional[str] = None
    date_column: Optional[str] = None
    categorical: List[str] = field(
        default_factory=lambda: ["home_team", "away_team", "divisional_game"]
    )
    numeric: List[str] = field(
        default_factory=lambda: [
            "season",
            "week",
            "home_moneyline",
            "away_moneyline",
            "home_spread",
            "away_spread",
            "home_team_prev_win",
            "away_team_prev_win",
            "home_rest",
            "away_rest",
            "temp",
            "wind",
            "elo_home_pre",
            "elo_away_pre",
            "elo_prob_home",
            "qbelo_prob_home",
        ]
    )
    odds: Optional[str] = "home_moneyline"


@dataclass
class SplitConfig:
    """Train/test splitting behaviour."""

    test_size: float = 0.2
    shuffle: bool = True
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model hyper-parameters."""

    estimator: str = "random_forest"
    random_state: int = 42
    n_estimators: int = 500
    max_depth: Optional[int] = None
    learning_rate: float = 0.05  # used by gradient boosting variants


@dataclass
class BettingConfig:
    """Value betting thresholds."""

    min_edge: float = 0.02  # 2% edge threshold
    min_probability: float = 0.05
    stake_size: float = 1.0


@dataclass
class PipelineConfig:
    """Aggregate pipeline configuration."""

    columns: ColumnConfig = field(default_factory=ColumnConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PipelineConfig":
        """Create a PipelineConfig from a raw dictionary (e.g., YAML)."""
        def build(name: str, dataclass_type):
            section = raw.get(name, {})
            return dataclass_type(**section)

        return cls(
            columns=build("columns", ColumnConfig),
            split=build("split", SplitConfig),
            model=build("model", ModelConfig),
            betting=build("betting", BettingConfig),
        )


def load_config(path: Optional[Path]) -> PipelineConfig:
    """Load configuration from YAML; fall back to defaults."""
    if path is None:
        return PipelineConfig()
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        return PipelineConfig()
    return PipelineConfig.from_dict(data)


__all__ = [
    "ColumnConfig",
    "SplitConfig",
    "ModelConfig",
    "BettingConfig",
    "PipelineConfig",
    "load_config",
]
