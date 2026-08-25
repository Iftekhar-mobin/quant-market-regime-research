"""Classification diagnostics.

Accuracy is reported here because reviewers expect it, and then largely ignored.
On a three-class directional target where "flat" is the majority, a model can
reach 60% accuracy by never taking a position; the numbers that carry
information are the per-class precision (what fraction of the positions taken
were right) and the economic value of those positions, which lives in the
backtest rather than here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)

CLASS_ORDER = [-1, 0, 1]
CLASS_NAMES = {-1: "Short", 0: "Flat", 1: "Long"}


def confusion_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Confusion matrix with readable row and column labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    names = [CLASS_NAMES[c] for c in CLASS_ORDER]
    frame = pd.DataFrame(matrix, index=names, columns=names)
    frame.index.name = "Actual"
    frame.columns.name = "Predicted"
    return frame


def classification_summary(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Headline classification metrics for a directional prediction."""
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    if y_true.empty:
        return {}

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Matthews correlation stays honest under class imbalance, which is why
        # it is the headline number rather than accuracy.
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }

    for label in CLASS_ORDER:
        name = CLASS_NAMES[label].lower()
        summary[f"{name}_precision"] = float(report[CLASS_NAMES[label]]["precision"])
        summary[f"{name}_recall"] = float(report[CLASS_NAMES[label]]["recall"])
        summary[f"{name}_support"] = float(report[CLASS_NAMES[label]]["support"])

    # Directional precision on the bars where a position was actually taken:
    # the single most decision-relevant classification number in the study.
    traded = y_pred != 0
    summary["signals_taken"] = float(traded.sum())
    summary["signal_rate"] = float(traded.mean())
    summary["directional_precision"] = (
        float((y_true[traded] == y_pred[traded]).mean()) if traded.any() else float("nan")
    )
    # Counting a short called as a long is a far worse error than a missed call.
    summary["sign_error_rate"] = (
        float(((y_true[traded] != 0) & (y_true[traded] == -y_pred[traded])).mean())
        if traded.any()
        else float("nan")
    )
    return summary


def per_class_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Precision / recall / F1 / support, one row per class."""
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in CLASS_ORDER:
        name = CLASS_NAMES[label]
        entry = report[name]
        rows.append(
            {
                "Class": name,
                "Precision": round(entry["precision"], 4),
                "Recall": round(entry["recall"], 4),
                "F1": round(entry["f1-score"], 4),
                "Support": int(entry["support"]),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    probabilities: pd.DataFrame,
    y_true: pd.Series,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """How precision and coverage trade off as the decision threshold rises.

    This is the diagnostic behind the choice of ``model.decision_threshold``: it
    shows directly how much selectivity has to be bought to reach a given
    directional precision, and how few positions are left at that price.
    """
    thresholds = thresholds if thresholds is not None else np.arange(0.34, 0.81, 0.02)
    y_true = y_true.reindex(probabilities.index)

    rows = []
    for threshold in thresholds:
        long_hit = probabilities["long"] >= threshold
        short_hit = probabilities["short"] >= threshold

        predicted = pd.Series(0, index=probabilities.index, dtype=int)
        predicted[long_hit & (probabilities["long"] >= probabilities["short"])] = 1
        predicted[short_hit & (probabilities["short"] > probabilities["long"])] = -1

        traded = predicted != 0
        rows.append(
            {
                "threshold": round(float(threshold), 3),
                "signal_rate": float(traded.mean()),
                "signals": int(traded.sum()),
                "directional_precision": float((y_true[traded] == predicted[traded]).mean())
                if traded.any()
                else np.nan,
                "sign_error_rate": float(
                    ((y_true[traded] != 0) & (y_true[traded] == -predicted[traded])).mean()
                )
                if traded.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
