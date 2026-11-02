"""LightGBM dataset, multiclass diagnostics, and probability calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

Label = Literal["down", "flat", "up"]
LABELS: tuple[Label, Label, Label] = ("down", "flat", "up")
LABEL_TO_INT = {label: index for index, label in enumerate(LABELS)}
FLAT_RETURN_BAND = 0.001
PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

_FLOW_COLUMNS = (
    "delta_15m", "delta_1h", "delta_6h", "cvd_change_15m", "cvd_change_1h", "cvd_change_6h",
    "imbalance_15m", "imbalance_1h", "imbalance_6h", "cls5_cls6_count_ratio_2h",
    "cls5_cls6_count_ratio_6h", "cls5_cls6_intensity_ratio_2h", "cls5_cls6_intensity_ratio_6h",
    "count_log_odds_2h", "count_log_odds_6h", "absorption_log_odds_2h", "absorption_log_odds_6h",
    "sweep_buy_1h", "sweep_sell_1h", "cvd_slope_pct", "bullish_delta_price_divergence",
    "bearish_delta_price_divergence",
)
FEATURE_COLUMNS = _FLOW_COLUMNS + (
    "distance_ema9", "distance_bb_mid", "return_15m", "return_1h", "atr_pct", "rsi3",
    "bb_width_pct", "realized_vol_pct", "trend_5m", "trend_15m", "trend_1h", "trend_4h",
    "band_score_15m",
)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Pooled causal features and three-class labels."""

    x: pd.DataFrame
    y: pd.Series


def model_frame(features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Create one 15m model row per completed bar without future fields."""

    execution = features["15m"]
    flow = features["flow"]
    frame = flow.reindex(execution.index).copy()
    close = execution["c"]
    frame["distance_ema9"] = (close - execution["ema9"]) / close
    frame["distance_bb_mid"] = (close - execution["bb_mid"]) / close
    frame["return_15m"] = execution["roc_1"]
    frame["return_1h"] = features["1h"]["roc_1"].reindex(execution.index).ffill().fillna(0.0)
    frame["atr_pct"] = execution["atr14"] / close
    frame["rsi3"] = execution["rsi3"]
    frame["bb_width_pct"] = execution["bb_width_pct"]
    frame["realized_vol_pct"] = execution["realized_vol_pct"]
    for timeframe in ("5m", "15m", "1h", "4h"):
        trend = features[timeframe]["trend"].reindex(execution.index).ffill().fillna("neutral")
        frame[f"trend_{timeframe}"] = trend.map({"bull": 1, "neutral": 0, "bear": -1}).fillna(0)
    frame["band_score_15m"] = execution["band_score"] / 4
    frame["close"] = close
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_dataset(
    features: Mapping[str, Mapping[str, pd.DataFrame]],
    start: datetime,
    end: datetime,
) -> Dataset:
    """Pool symbols and drop labels unavailable or crossing the train cutoff."""

    xs = []
    ys = []
    for symbol_frames in features.values():
        frame = model_frame(symbol_frames)
        forward = frame["close"].shift(-4)
        returns = forward / frame["close"] - 1
        available = (frame.index >= start) & (frame.index + timedelta(hours=1) <= end)
        x = frame.loc[available, list(FEATURE_COLUMNS)].astype("float64")
        forward_returns = returns.loc[available]
        y = pd.Series(
            np.select(
                [forward_returns < -FLAT_RETURN_BAND, forward_returns > FLAT_RETURN_BAND],
                [0, 2],
                default=1,
            ),
            index=x.index,
            dtype="int64",
        )
        xs.append(x)
        ys.append(y)
    if not xs:
        return Dataset(pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype="int64"))
    return Dataset(pd.concat(xs), pd.concat(ys))


class LearnedFlowModel:
    """Deterministic native LightGBM Booster with gain importances."""

    def __init__(self, seed: int = 20261102) -> None:
        self.seed = seed
        self._model: lgb.Booster | None = None
        self.feature_names: tuple[str, ...] = ()

    def fit(self, dataset: Dataset) -> None:
        if dataset.x.empty or dataset.y.nunique() < 3:
            raise ValueError("training requires all three label classes")
        self.feature_names = tuple(dataset.x.columns)
        counts = dataset.y.value_counts()
        weights = dataset.y.map(len(dataset.y) / (3 * counts))
        train_set = lgb.Dataset(dataset.x, label=dataset.y, weight=weights)
        parameters = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 100,
            "verbosity": -1,
            "seed": self.seed,
            "feature_pre_filter": False,
        }
        self._model = lgb.train(parameters, train_set, num_boost_round=120)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("model is not fitted")
        values = np.asarray(self._model.predict(x.loc[:, self.feature_names]), dtype=float)
        return values

    def feature_importances(self) -> list[dict[str, float | str]]:
        if self._model is None:
            raise ValueError("model is not fitted")
        gains = self._model.feature_importance(importance_type="gain")
        total = float(np.sum(gains))
        rows: list[dict[str, float | str]] = [
            {"feature": name, "gain": float(gain / total) if total > 0 else 0.0}
            for name, gain in zip(self.feature_names, gains, strict=True)
        ]
        return sorted(rows, key=lambda row: float(row["gain"]), reverse=True)[:10]


def probability_frame(
    model: LearnedFlowModel,
    features: Mapping[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Predict causal rows in `[start, end)` with down/flat/up columns."""

    frame = model_frame(features)
    selected = frame.loc[(frame.index >= start) & (frame.index < end), list(FEATURE_COLUMNS)]
    probabilities = model.predict_proba(selected)
    return pd.DataFrame(probabilities, columns=LABELS, index=selected.index)


def calibrate_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    """Select probability by train-only macro-F1 on the flat-excluded task."""

    if len(y) != len(probabilities):
        raise ValueError("label/probability length mismatch")
    best_score = float("-inf")
    best_threshold = 0.50
    for threshold in PROBABILITY_THRESHOLDS:
        predicted, actual, _ = directional_predictions(y, probabilities, threshold)
        score = macro_f1(predicted, actual) if len(predicted) else 0.0
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def directional_predictions(
    y: pd.Series, probabilities: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return predicted class, actual class, and up probabilities for confident rows."""

    up = probabilities[:, 2]
    down = probabilities[:, 0]
    confidence = np.maximum(up, down)
    mask = (confidence > threshold) & (y.to_numpy() != 1)
    predicted = (up[mask] > down[mask]).astype("int64")
    actual = (y.to_numpy()[mask] == 2).astype("int64")
    normalized_up = up[mask] / np.maximum(up[mask] + down[mask], 1e-12)
    return predicted, actual, normalized_up


def diagnostics(y: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute flat-excluded accuracy/AUC and a five-bin calibration summary."""

    predicted, actual, up_probability = directional_predictions(y, probabilities, 0.0)
    accuracy = float(np.mean(predicted == actual)) if len(actual) else 0.0
    auc = binary_auc(actual, up_probability)
    calibration = []
    for lower, upper in zip((0.0, 0.2, 0.4, 0.6, 0.8), (0.2, 0.4, 0.6, 0.8, 1.0), strict=True):
        mask = (up_probability >= lower) & (up_probability < upper if upper < 1 else up_probability <= upper)
        count = int(mask.sum())
        calibration.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": count,
                "mean_up_probability": float(np.mean(up_probability[mask])) if count else 0.0,
                "empirical_up_rate": float(np.mean(actual[mask])) if count else 0.0,
            }
        )
    return {"flat_excluded_rows": len(actual), "accuracy": accuracy, "auc": auc, "calibration": calibration}


def binary_auc(actual: np.ndarray, score: np.ndarray) -> float:
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def macro_f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    scores = []
    for label in (0, 1):
        true_positive = int(np.sum((predicted == label) & (actual == label)))
        false_positive = int(np.sum((predicted == label) & (actual != label)))
        false_negative = int(np.sum((predicted != label) & (actual == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))
