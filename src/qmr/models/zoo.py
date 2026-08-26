"""Directional learners.

Every model in the registry is wrapped in :class:`DirectionalModel`, which
presents one interface to the rest of the pipeline:

    model.fit(X, y)          # y in {-1, 0, +1}
    model.predict_proba(X)   # DataFrame with columns short / flat / long

That uniformity is what lets the experiment runner compare a logistic regression
against a gradient-boosted ensemble against a recurrent network without any
branching outside this module — which is the point of a comparison study.

Feature scaling is folded into each pipeline, fitted on the training window
only. Scaling on the full sample is a subtle leak: the mean and variance of the
test period are not knowable while the model is being fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from qmr.config import ModelConfig
from qmr.logging_utils import get_logger

log = get_logger(__name__)

# Internal class order. The pipeline speaks in -1 / 0 / +1; scikit-learn and the
# boosting libraries want contiguous non-negative integers.
CLASS_ORDER = [-1, 0, 1]
CLASS_NAMES = {-1: "short", 0: "flat", 1: "long"}
PROBABILITY_COLUMNS = ["short", "flat", "long"]


@dataclass
class ModelSpec:
    """Registry entry: how to build a model and how to describe it."""

    key: str
    label: str
    family: str
    description: str
    defaults: dict[str, Any] = field(default_factory=dict)
    requires: str | None = None  # optional dependency, checked lazily


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logistic": ModelSpec(
        key="logistic",
        label="Logistic regression",
        family="Linear",
        description=(
            "Regularised multinomial logit. The control arm: any tree ensemble that "
            "cannot beat it is buying complexity with nothing in return."
        ),
        defaults={"C": 0.1, "max_iter": 2000},
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random forest",
        family="Bagged trees",
        description=(
            "Bagged decision trees. Low variance and hard to overfit, but averages "
            "away the sharp interactions that carry most of the signal here."
        ),
        defaults={"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 50},
    ),
    "hist_gradient_boosting": ModelSpec(
        key="hist_gradient_boosting",
        label="Histogram gradient boosting",
        family="Boosted trees",
        description=(
            "Scikit-learn native boosting. No extra dependency, and a useful check "
            "that a result is not an artefact of one particular boosting library."
        ),
        defaults={"max_iter": 400, "learning_rate": 0.05, "max_depth": 6},
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        label="XGBoost",
        family="Boosted trees",
        description=(
            "Gradient-boosted trees with strong regularisation. The reference "
            "learner for tabular financial features."
        ),
        defaults={
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 20,
            "reg_lambda": 2.0,
        },
        requires="xgboost",
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        label="LightGBM",
        family="Boosted trees",
        description=(
            "Leaf-wise boosting. Considerably faster than XGBoost on wide feature "
            "matrices, at the cost of a stronger tendency to overfit shallow data."
        ),
        defaults={
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        },
        requires="lightgbm",
    ),
    "mlp": ModelSpec(
        key="mlp",
        label="Neural network (MLP)",
        family="Neural",
        description=(
            "Feed-forward network on the same tabular features. Tests whether the "
            "structure the trees find is genuinely non-linear."
        ),
        defaults={"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "max_iter": 400},
    ),
    "lstm": ModelSpec(
        key="lstm",
        label="LSTM sequence model",
        family="Recurrent",
        description=(
            "Recurrent network over a rolling window of feature vectors. Sees the "
            "path rather than a snapshot, which is the one thing the tabular models "
            "structurally cannot do."
        ),
        defaults={"lookback": 24, "units": 64, "dropout": 0.2, "epochs": 12, "batch_size": 128},
        requires="tensorflow",
    ),
    "gru": ModelSpec(
        key="gru",
        label="GRU sequence model",
        family="Recurrent",
        description=(
            "Gated recurrent unit; the same idea as the LSTM with fewer parameters, "
            "which usually helps on samples this size."
        ),
        defaults={"lookback": 24, "units": 64, "dropout": 0.2, "epochs": 12, "batch_size": 128},
        requires="tensorflow",
    ),
}

SEQUENCE_MODELS = {"lstm", "gru"}


def model_catalogue() -> pd.DataFrame:
    """The registry as a table, for the research console."""
    return pd.DataFrame(
        [
            {
                "Key": spec.key,
                "Model": spec.label,
                "Family": spec.family,
                "Available": is_available(spec.key),
                "What it tests": spec.description,
            }
            for spec in MODEL_REGISTRY.values()
        ]
    )


def is_available(key: str) -> bool:
    """Whether the optional dependency for a model is importable."""
    spec = MODEL_REGISTRY.get(key)
    if spec is None or spec.requires is None:
        return True
    try:
        __import__(spec.requires)
    except ImportError:
        return False
    return True


def available_models() -> list[str]:
    return [key for key in MODEL_REGISTRY if is_available(key)]


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------
def _tabular_estimator(key: str, params: dict[str, Any], seed: int):
    if key == "logistic":
        return LogisticRegression(random_state=seed, **params)
    if key == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if key == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    if key == "mlp":
        return MLPClassifier(random_state=seed, **params)
    if key == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            n_jobs=-1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            **params,
        )
    if key == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, n_jobs=-1, objective="multiclass", **params)

    raise ValueError(f"No tabular estimator for {key!r}")


class _SequenceEstimator:
    """Minimal recurrent classifier with a scikit-learn shaped interface.

    Kept deliberately small: the study compares *representations* (a snapshot of
    engineered features versus a window of them), not architectures, so the
    network is a single recurrent layer over a rolling window and nothing more.
    """

    def __init__(self, kind: str, params: dict[str, Any], seed: int) -> None:
        self.kind = kind
        self.lookback = int(params.pop("lookback", 24))
        self.units = int(params.pop("units", 64))
        self.dropout = float(params.pop("dropout", 0.2))
        self.epochs = int(params.pop("epochs", 12))
        self.batch_size = int(params.pop("batch_size", 128))
        self.seed = seed
        self._model = None
        self._n_classes = 3

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(self.seed)
        recurrent = layers.LSTM if self.kind == "lstm" else layers.GRU

        model = models.Sequential(
            [
                layers.Input(shape=(self.lookback, n_features)),
                recurrent(self.units, return_sequences=False),
                layers.Dropout(self.dropout),
                layers.Dense(32, activation="relu"),
                layers.Dense(self._n_classes, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _windows(self, matrix: np.ndarray) -> np.ndarray:
        """Rolling windows ending at each row, padded at the head by repetition."""
        n_rows, n_features = matrix.shape
        padded = np.vstack([np.repeat(matrix[:1], self.lookback - 1, axis=0), matrix])
        windows = np.empty((n_rows, self.lookback, n_features), dtype=np.float32)
        for i in range(n_rows):
            windows[i] = padded[i : i + self.lookback]
        return windows

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        import os

        # Keras is noisy on import and during training; the study log is the
        # record that matters here.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        windows = self._windows(np.asarray(X, dtype=np.float32))
        self._model = self._build(windows.shape[2])
        self._model.fit(
            windows,
            np.asarray(y),
            epochs=self.epochs,
            batch_size=self.batch_size,
            sample_weight=sample_weight,
            verbose=0,
            shuffle=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Sequence model has not been fitted.")
        windows = self._windows(np.asarray(X, dtype=np.float32))
        return self._model.predict(windows, batch_size=512, verbose=0)


# ---------------------------------------------------------------------------
# Uniform wrapper
# ---------------------------------------------------------------------------
class DirectionalModel:
    """A three-class directional classifier over engineered features."""

    def __init__(self, config: ModelConfig, seed: int = 7) -> None:
        self.config = config
        self.seed = seed
        self.spec = MODEL_REGISTRY.get(config.name)
        if self.spec is None:
            raise ValueError(
                f"Unknown model {config.name!r}. Available: {sorted(MODEL_REGISTRY)}"
            )
        if not is_available(config.name):
            raise ImportError(
                f"Model {self.spec.label} needs the optional dependency "
                f"{self.spec.requires!r}. Install it with: pip install {self.spec.requires}"
            )

        self.params = {**self.spec.defaults, **(config.params or {})}
        self.feature_names_: list[str] = []
        self.selected_features_: list[str] = []
        self.classes_: list[int] = []
        self._estimator = None

    @property
    def is_sequence_model(self) -> bool:
        return self.config.name in SEQUENCE_MODELS

    @property
    def label(self) -> str:
        return self.spec.label

    # -- training ---------------------------------------------------------
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> list[str]:
        """Rank features on this training fold and keep the strongest k.

        Eighty features over a signal this weak is mostly noise, and every
        useless column is one more chance for a tree to split on an accident.
        The ranking is fitted on the training window only, like everything else,
        so the selection cannot see the test period.
        """
        k = self.config.top_k_features
        if not k or k >= X.shape[1]:
            return list(X.columns)

        from sklearn.ensemble import RandomForestClassifier

        probe = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=50,
            random_state=self.seed,
            n_jobs=-1,
        )
        probe.fit(X.fillna(X.median(numeric_only=True)).fillna(0.0), y)
        ranked = pd.Series(probe.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        chosen = ranked.head(k).index.tolist()
        log.info("Selected %d of %d features; top 3: %s", k, X.shape[1], chosen[:3])
        return chosen

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DirectionalModel:
        self.selected_features_ = self._select_features(X, y)
        X = X[self.selected_features_]
        self.feature_names_ = list(X.columns)
        encoded = y.map({value: position for position, value in enumerate(CLASS_ORDER)})

        if encoded.isna().any():
            bad = sorted(set(y.unique()) - set(CLASS_ORDER))
            raise ValueError(f"Labels must be in {CLASS_ORDER}; found {bad}")
        encoded = encoded.to_numpy(dtype=int)
        self.classes_ = CLASS_ORDER

        sample_weight = None
        if self.config.class_balance:
            # Flat dominates most label sets; without reweighting the model
            # learns to predict "do nothing" and reports a fine accuracy.
            sample_weight = compute_sample_weight("balanced", encoded)

        if self.is_sequence_model:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(
                np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            )
            estimator = _SequenceEstimator(self.config.name, dict(self.params), self.seed)
            estimator.fit(matrix, encoded, sample_weight=sample_weight)
            self._scaler = scaler
            self._estimator = estimator
            return self

        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", _tabular_estimator(self.config.name, dict(self.params), self.seed)),
        ]
        # Tree ensembles neither need nor benefit from standardisation.
        if self.config.name in {"random_forest", "hist_gradient_boosting", "xgboost", "lightgbm"}:
            steps = [step for step in steps if step[0] != "scale"]

        pipeline = Pipeline(steps)
        fit_params = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
        pipeline.fit(X, encoded, **fit_params)
        self._estimator = pipeline
        return self

    # -- inference --------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted.")

        # Apply the same column selection that fit() used.
        X = X[self.selected_features_]

        if self.is_sequence_model:
            matrix = self._scaler.transform(
                np.nan_to_num(X.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            )
            probabilities = self._estimator.predict_proba(matrix)
        else:
            probabilities = self._estimator.predict_proba(X)

        frame = pd.DataFrame(probabilities, index=X.index, columns=PROBABILITY_COLUMNS[: probabilities.shape[1]])
        # A fold whose training window happened to contain only two classes
        # returns two columns; fill the missing one with zero probability.
        for column in PROBABILITY_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        return frame[PROBABILITY_COLUMNS]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probabilities = self.predict_proba(X)
        winner = probabilities.to_numpy().argmax(axis=1)
        return pd.Series([CLASS_ORDER[i] for i in winner], index=X.index, name="prediction")

    # -- interpretation ---------------------------------------------------
    def feature_importance(self) -> pd.Series | None:
        """Model-native importance, normalised to sum to one.

        Returns ``None`` for models that do not expose one (the recurrent
        networks), rather than fabricating a surrogate.
        """
        if self._estimator is None or self.is_sequence_model:
            return None

        model = self._estimator.named_steps.get("model")
        importance = getattr(model, "feature_importances_", None)

        if importance is None:
            coefficients = getattr(model, "coef_", None)
            if coefficients is None:
                return None
            importance = np.abs(coefficients).mean(axis=0)

        if len(importance) != len(self.feature_names_):
            return None

        series = pd.Series(importance, index=self.feature_names_, dtype=float)
        total = series.sum()
        return (series / total if total > 0 else series).sort_values(ascending=False)


class SeedEnsemble:
    """Average the probabilities of several models that differ only by seed.

    A single tree ensemble on a weak signal produces probabilities that wobble
    near the decision threshold for reasons that have nothing to do with the
    market. Every wobble across the threshold is a position flip, and every flip
    pays the spread. Averaging over seeds damps the wobble without changing the
    expected prediction, so it buys lower turnover at no cost in accuracy.

    It is the cheapest variance reduction available here, which matters because
    this study is limited by cost per unit of signal rather than by signal.
    """

    def __init__(self, config: ModelConfig, seed: int = 7) -> None:
        self.config = config
        self.members = [
            DirectionalModel(config, seed=seed + offset) for offset in range(config.n_seeds)
        ]

    @property
    def label(self) -> str:
        return f"{self.members[0].label} x{len(self.members)} seeds"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeedEnsemble:
        for member in self.members:
            member.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        total = None
        for member in self.members:
            probabilities = member.predict_proba(X)
            total = probabilities if total is None else total + probabilities
        return total / len(self.members)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        winner = self.predict_proba(X).to_numpy().argmax(axis=1)
        return pd.Series([CLASS_ORDER[i] for i in winner], index=X.index, name="prediction")

    def feature_importance(self) -> pd.Series | None:
        parts = [m.feature_importance() for m in self.members]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).mean(axis=1).sort_values(ascending=False)


def build_model(config: ModelConfig, seed: int = 7):
    if config.n_seeds > 1:
        return SeedEnsemble(config, seed=seed)
    return DirectionalModel(config, seed=seed)
