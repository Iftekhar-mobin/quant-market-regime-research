"""Market regime detectors.

A regime is a persistent state of the market — trending or ranging, calm or
stressed — that changes what a strategy should expect from its own signals. The
detectors here all share one interface:

    detector.fit(descriptors)      # training window only
    detector.predict(descriptors)  # any window

Keeping ``fit`` and ``predict`` separate is not decoration. Fitting a clustering
model on the full sample and then evaluating a strategy "out of sample" inside
those clusters leaks the whole future into the regime assignment; the
walk-forward loop therefore refits the detector on each training window and only
ever *applies* it to the corresponding test window.

Detectors operate on the four regime descriptors emitted by the feature
pipeline (trend strength, volatility percentile, momentum, mean-reversion
tendency) rather than on the full feature matrix, which is what makes the
resulting states interpretable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from qmr.config import RegimeConfig
from qmr.logging_utils import get_logger

log = get_logger(__name__)


class RegimeDetector(ABC):
    """Common interface for every regime detector."""

    name: str = "base"

    def __init__(self, features: list[str], n_regimes: int, random_state: int = 7) -> None:
        self.features = list(features)
        self.n_regimes = int(n_regimes)
        self.random_state = int(random_state)
        self._fitted = False
        self.labels_: dict[int, str] = {}

    # -- helpers ----------------------------------------------------------
    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(
                f"Regime descriptor(s) {missing} missing from the feature frame. "
                f"Available: {sorted(frame.columns)[:20]}..."
            )
        return frame[self.features].to_numpy(dtype=float)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before predicting.")

    # -- interface --------------------------------------------------------
    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> RegimeDetector: ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.Series: ...

    def fit_predict(self, frame: pd.DataFrame) -> pd.Series:
        return self.fit(frame).predict(frame)

    def describe(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Mean descriptor values per regime, with the assigned plain-English name."""
        assignments = self.predict(frame)
        profile = frame.loc[assignments.index, self.features].groupby(assignments).mean()
        profile.insert(0, "name", [self.labels_.get(int(i), f"Regime {i}") for i in profile.index])
        profile.insert(1, "share", assignments.value_counts(normalize=True).sort_index().to_numpy())
        return profile

    # -- naming -----------------------------------------------------------
    def _assign_names(self, centres: pd.DataFrame) -> None:
        """Turn cluster centroids into readable names.

        The names are derived from where a centroid sits relative to the others
        on the trend and volatility axes, so they stay meaningful no matter how
        the clustering happens to order its labels between refits.
        """
        if "trend_strength" not in centres.columns:
            self.labels_ = {int(i): f"Regime {i}" for i in centres.index}
            return

        trend = centres["trend_strength"]
        volatility = centres.get("vol_percentile", pd.Series(0.5, index=centres.index))

        trend_cut = trend.abs().median()
        vol_cut = volatility.median()

        names: dict[int, str] = {}
        for regime in centres.index:
            direction = (
                "Uptrend"
                if trend[regime] > trend_cut
                else "Downtrend"
                if trend[regime] < -trend_cut
                else "Range"
            )
            intensity = "high-volatility" if volatility[regime] > vol_cut else "quiet"
            names[int(regime)] = f"{direction}, {intensity}"

        # Two centroids can legitimately land on the same description; keep the
        # names unique so they can be used as chart legends and column keys.
        seen: dict[str, int] = {}
        for regime, label in names.items():
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:
                names[regime] = f"{label} ({seen[label]})"

        self.labels_ = names


class KMeansRegimeDetector(RegimeDetector):
    """Hard partition of the descriptor space into ``n_regimes`` states."""

    name = "kmeans"

    def fit(self, frame: pd.DataFrame) -> KMeansRegimeDetector:
        matrix = self._matrix(frame)
        self._scaler = StandardScaler().fit(matrix)
        self._model = KMeans(
            n_clusters=self.n_regimes,
            n_init=10,
            random_state=self.random_state,
        ).fit(self._scaler.transform(matrix))

        centres = pd.DataFrame(
            self._scaler.inverse_transform(self._model.cluster_centers_),
            columns=self.features,
        )
        self._assign_names(centres)
        self._fitted = True
        log.info("Fitted k-means regimes on %d bars: %s", len(frame), list(self.labels_.values()))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        matrix = self._scaler.transform(self._matrix(frame))
        return pd.Series(self._model.predict(matrix), index=frame.index, name="regime")


class GaussianMixtureRegimeDetector(RegimeDetector):
    """Soft partition; each bar carries a posterior over the regimes.

    Preferred over k-means when the states overlap, which they usually do around
    transitions — a market does not switch from "quiet range" to "volatile
    trend" between one bar and the next.
    """

    name = "gmm"

    def fit(self, frame: pd.DataFrame) -> GaussianMixtureRegimeDetector:
        matrix = self._matrix(frame)
        self._scaler = StandardScaler().fit(matrix)
        self._model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=self.random_state,
        ).fit(self._scaler.transform(matrix))

        centres = pd.DataFrame(
            self._scaler.inverse_transform(self._model.means_), columns=self.features
        )
        self._assign_names(centres)
        self._fitted = True
        log.info("Fitted Gaussian-mixture regimes on %d bars", len(frame))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        matrix = self._scaler.transform(self._matrix(frame))
        return pd.Series(self._model.predict(matrix), index=frame.index, name="regime")

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        matrix = self._scaler.transform(self._matrix(frame))
        return pd.DataFrame(
            self._model.predict_proba(matrix),
            index=frame.index,
            columns=[self.labels_.get(i, f"Regime {i}") for i in range(self.n_regimes)],
        )


class RuleBasedRegimeDetector(RegimeDetector):
    """Four fixed quadrants: trend direction crossed with volatility level.

    This is the transparent benchmark the learned detectors have to beat. If a
    clustering model cannot outperform "is the market trending, and is it
    volatile", the extra machinery is not earning its place in the study.

    The thresholds are quantiles of the *training* descriptors, so the detector
    is still fitted — it simply has two parameters instead of a full covariance
    structure.
    """

    name = "rule"

    def __init__(self, features: list[str], n_regimes: int = 4, random_state: int = 7) -> None:
        super().__init__(features, 4, random_state)  # the rule defines exactly four states
        self.labels_ = {
            0: "Range, quiet",
            1: "Range, high-volatility",
            2: "Trend, quiet",
            3: "Trend, high-volatility",
        }

    def fit(self, frame: pd.DataFrame) -> RuleBasedRegimeDetector:
        trend = frame["trend_strength"].abs()
        volatility = frame["vol_percentile"]
        self._trend_cut = float(trend.quantile(0.5))
        self._vol_cut = float(volatility.quantile(0.5))
        self._fitted = True
        log.info(
            "Fitted rule-based regimes: |trend| cut %.3f, volatility cut %.3f",
            self._trend_cut,
            self._vol_cut,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        is_trending = (frame["trend_strength"].abs() > self._trend_cut).to_numpy()
        is_volatile = (frame["vol_percentile"] > self._vol_cut).to_numpy()
        codes = is_trending.astype(int) * 2 + is_volatile.astype(int)
        return pd.Series(codes, index=frame.index, name="regime")


class NullRegimeDetector(RegimeDetector):
    """Single-regime placeholder, used for the no-regime control arm."""

    name = "none"

    def __init__(self, features: list[str], n_regimes: int = 1, random_state: int = 7) -> None:
        super().__init__(features, 1, random_state)
        self.labels_ = {0: "All market conditions"}

    def fit(self, frame: pd.DataFrame) -> NullRegimeDetector:
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        return pd.Series(0, index=frame.index, name="regime")


DETECTORS: dict[str, type[RegimeDetector]] = {
    "none": NullRegimeDetector,
    "rule": RuleBasedRegimeDetector,
    "kmeans": KMeansRegimeDetector,
    "gmm": GaussianMixtureRegimeDetector,
}


def build_detector(config: RegimeConfig, random_state: int = 7) -> RegimeDetector:
    """Instantiate the detector named in the configuration."""
    try:
        detector_cls = DETECTORS[config.method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown regime method {config.method!r}. Available: {sorted(DETECTORS)}"
        ) from exc
    return detector_cls(
        features=config.features, n_regimes=config.n_regimes, random_state=random_state
    )
