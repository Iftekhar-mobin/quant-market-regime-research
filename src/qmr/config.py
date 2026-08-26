"""Typed experiment configuration.

The configuration is a plain nested dataclass tree loaded from YAML. Keeping it
typed (rather than passing dictionaries around) means a misspelled key fails at
load time instead of silently changing the meaning of a study.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qmr.paths import DEFAULT_CONFIG_PATH


@dataclass
class ExperimentMeta:
    name: str = "baseline"
    seed: int = 7
    notes: str = ""


@dataclass
class DataConfig:
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    start: str | None = None
    end: str | None = None
    warmup_bars: int = 300


@dataclass
class FeatureConfig:
    blocks: list[str] = field(
        default_factory=lambda: [
            "returns",
            "trend",
            "momentum",
            "volatility",
            "structure",
            "volume",
            "session",
        ]
    )
    return_horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])
    volatility_windows: list[int] = field(default_factory=lambda: [14, 50, 200])
    trend_windows: list[int] = field(default_factory=lambda: [20, 50, 200])


@dataclass
class LabelConfig:
    method: str = "triple_barrier"
    horizon: int = 24
    atr_window: int = 14
    take_profit_atr: float = 1.0
    stop_loss_atr: float = 1.0
    deadband_atr: float = 0.25
    # `meta` only: the rule-based strategy that supplies the direction. The
    # learner then decides which of its trades are worth taking.
    primary: str = "ma_crossover"


@dataclass
class RegimeConfig:
    method: str = "kmeans"
    n_regimes: int = 4
    features: list[str] = field(
        default_factory=lambda: [
            "trend_strength",
            "vol_percentile",
            "momentum_score",
            "mean_reversion_score",
        ]
    )
    as_model_feature: bool = True
    specialised_models: bool = False


@dataclass
class ModelConfig:
    name: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)
    decision_threshold: float = 0.50
    class_balance: bool = True
    # Keep only the k most important features, ranked on the training fold.
    # None keeps all of them.
    top_k_features: int | None = None
    # Average the probabilities of this many models, differing only by seed.
    n_seeds: int = 1


@dataclass
class ValidationConfig:
    scheme: str = "expanding"
    n_folds: int = 6
    test_size: float = 0.12
    min_train_size: float = 0.25
    embargo_bars: int = 48


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    cost_bps: float = 2.0
    slippage_bps: float = 0.5
    position_size: float = 1.0
    execution_lag: int = 1
    min_holding_bars: int = 24
    bars_per_year: int = 6240
    # Annualised volatility to size positions towards. None disables sizing and
    # every position is taken at full size.
    volatility_target: float | None = None
    volatility_window: int = 100
    max_leverage: float = 3.0
    # Restrict new entries to these hours (exchange/UTC hour of the bar). Empty
    # means trade around the clock.
    session_hours: list[int] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class Config:
    experiment: ExperimentMeta = field(default_factory=ExperimentMeta)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")

    def copy(self) -> Config:
        return Config.from_dict(copy.deepcopy(self.to_dict()))

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        sections = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in (payload or {}).items():
            if key not in sections:
                raise KeyError(
                    f"Unknown configuration section '{key}'. "
                    f"Valid sections: {sorted(sections)}"
                )
            kwargs[key] = _build_section(cls, key, value or {})
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-path overrides applied.

        >>> cfg.with_overrides({"model.name": "lightgbm", "regime.n_regimes": 3})
        """
        payload = self.to_dict()
        for dotted, value in overrides.items():
            target = payload
            *parents, leaf = dotted.split(".")
            for part in parents:
                if part not in target:
                    raise KeyError(f"Unknown configuration path: {dotted}")
                target = target[part]
            if leaf not in target:
                raise KeyError(f"Unknown configuration key: {dotted}")
            target[leaf] = value
        return Config.from_dict(payload)


_SECTION_TYPES: dict[str, type] = {
    "experiment": ExperimentMeta,
    "data": DataConfig,
    "features": FeatureConfig,
    "labeling": LabelConfig,
    "regime": RegimeConfig,
    "model": ModelConfig,
    "validation": ValidationConfig,
    "backtest": BacktestConfig,
    "evaluation": EvaluationConfig,
}


def _build_section(_owner: type, key: str, value: dict[str, Any]):
    section_cls = _SECTION_TYPES[key]
    valid = {f.name for f in dataclasses.fields(section_cls)}
    unknown = set(value) - valid
    if unknown:
        raise KeyError(
            f"Unknown key(s) {sorted(unknown)} in section '{key}'. Valid keys: {sorted(valid)}"
        )
    return section_cls(**value)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key.path=value`` CLI override, typing the value via YAML rules."""
    if "=" not in text:
        raise ValueError(f"Override must look like key.path=value, got: {text!r}")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)
