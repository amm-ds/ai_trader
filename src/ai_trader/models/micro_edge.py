"""Purged LightGBM with OOF isotonic calibration and a causal cost gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ai_trader.evaluation.walk_forward import Fold
from ai_trader.models.learned_flow import Dataset, diagnostics, macro_f1

FLAT_BANDS = (0.0008, 0.0010, 0.0015)
THRESHOLD_QUANTILES = (0.60, 0.70, 0.80)
ROUND_TRIP_COST = 0.0022
FEATURE_COLUMNS = (
    "ofi_15m", "buy_pressure", "sell_pressure", "cvd_slope_1h", "microprice_deviation_ewma",
    "signed_volume_z", "dollar_volume_z", "trade_count_z", "sweep_buy", "sweep_sell",
    "return_15m", "return_1h", "atr_pct", "rsi3", "bb_width_pct", "realized_vol_pct",
    "bb_width_pct_1h", "bb_width_pct_4h", "realized_vol_pct_1h", "realized_vol_pct_4h",
    "distance_ema9", "distance_bb_mid", "trend_5m", "trend_15m", "trend_1h", "trend_4h",
    "band_score_15m",
)


@dataclass(frozen=True, slots=True)
class CalibratedModel:
    """Final primary Booster plus one isotonic calibrator per class."""

    booster: lgb.Booster
    calibrators: tuple[IsotonicRegression, IsotonicRegression, IsotonicRegression]
    feature_names: tuple[str, ...]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.booster.predict(x.loc[:, self.feature_names]), dtype=float)
        calibrated = np.column_stack(
            [calibrator.predict(raw[:, index]) for index, calibrator in enumerate(self.calibrators)]
        )
        calibrated = np.clip(calibrated, 0.0, 1.0)
        normalized = calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
        return np.asarray(normalized, dtype=float)

    def feature_importances(self) -> list[dict[str, float | str]]:
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(np.sum(gains))
        rows: list[dict[str, float | str]] = [
            {"feature": name, "gain": float(gain / total) if total > 0 else 0.0}
            for name, gain in zip(self.feature_names, gains, strict=True)
        ]
        return sorted(rows, key=lambda row: float(row["gain"]), reverse=True)[:10]


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen feature matrix and causal expected 1h move."""

    close = frame["c"]
    result = frame.copy()
    result["distance_ema9"] = (close - frame["ema9"]) / close
    result["distance_bb_mid"] = (close - frame["bb_mid"]) / close
    result["atr_pct"] = frame["atr14"] / close
    for timeframe in ("5m", "15m", "1h", "4h"):
        result[f"trend_{timeframe}"] = frame[f"trend_{timeframe}"].map({"bull": 1, "neutral": 0, "bear": -1}).fillna(0)
    result["band_score_15m"] = frame["band_score"] / 4
    result["expected_move_1h"] = close.pct_change(4).abs().rolling(96, min_periods=20).mean()
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def make_dataset(frame: pd.DataFrame, start: datetime, end: datetime, flat_band: float) -> Dataset:
    """Label 1h forward returns, excluding labels that cross the train cutoff."""

    modeled = model_frame(frame)
    returns = modeled["c"].shift(-4) / modeled["c"] - 1
    available = (modeled.index >= start) & (modeled.index + timedelta(hours=1) <= end)
    x = modeled.loc[available, list(FEATURE_COLUMNS)].astype("float64")
    forward = returns.loc[available]
    y = pd.Series(np.select([forward < -flat_band, forward > flat_band], [0, 2], default=1), index=x.index, dtype="int64")
    return Dataset(x, y)


def purged_folds(start: datetime, end: datetime) -> tuple[Fold, ...]:
    """Five expanding folds with a one-day embargo before each OOS segment."""

    segment = (end - start) / 8
    folds = []
    for index in range(5):
        test_start = start + segment * (index + 3)
        folds.append(
            Fold(index + 1, start, test_start - timedelta(days=1), test_start, test_start + segment)
        )
    return tuple(folds)


def fit_primary(dataset: Dataset, seed: int) -> lgb.Booster:
    if dataset.x.empty or dataset.y.nunique() < 3:
        raise ValueError("primary training requires all three classes")
    counts = dataset.y.value_counts()
    weights = dataset.y.map(len(dataset.y) / (3 * counts))
    parameters = {
        "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100,
        "verbosity": -1, "seed": seed, "feature_pre_filter": False,
    }
    return lgb.train(parameters, lgb.Dataset(dataset.x, label=dataset.y, weight=weights), num_boost_round=120)


def fit_calibrated(
    frame: pd.DataFrame,
    start: datetime,
    end: datetime,
    flat_band: float,
    seed: int = 20261102,
) -> tuple[CalibratedModel, pd.Series, np.ndarray]:
    """Fit three purged inner OOF models, isotonic maps, then the full primary."""

    duration = (end - start) / 4
    oof_y = []
    oof_probabilities = []
    for index in range(1, 4):
        validation_start = start + duration * index
        validation_end = start + duration * (index + 1)
        train = make_dataset(frame, start, validation_start - timedelta(days=1), flat_band)
        booster = fit_primary(train, seed + index)
        validation = make_dataset(frame, validation_start, validation_end, flat_band)
        probabilities = np.asarray(booster.predict(validation.x), dtype=float)
        oof_y.append(validation.y)
        oof_probabilities.append(probabilities)
    y = pd.concat(oof_y)
    probabilities = np.concatenate(oof_probabilities)
    calibrators = []
    for class_id in range(3):
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(probabilities[:, class_id], (y.to_numpy() == class_id).astype("int64"))
        calibrators.append(calibrator)
    final = fit_primary(make_dataset(frame, start, end, flat_band), seed)
    model = CalibratedModel(final, tuple(calibrators), tuple(FEATURE_COLUMNS))
    calibrated = np.column_stack(
        [calibrator.predict(probabilities[:, index]) for index, calibrator in enumerate(calibrators)]
    )
    calibrated = np.clip(calibrated, 0.0, 1.0)
    calibrated /= np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
    return model, y, calibrated


def threshold_from_oof(y: pd.Series, probabilities: np.ndarray, quantile: float) -> float:
    """Select a confidence quantile on purged OOF flat-excluded predictions."""

    confidence = np.maximum(probabilities[:, 0], probabilities[:, 2])
    mask = y.to_numpy() != 1
    return float(np.quantile(confidence[mask], quantile)) if mask.any() else 1.0


def oof_score(y: pd.Series, probabilities: np.ndarray, threshold: float) -> float:
    confidence = np.maximum(probabilities[:, 0], probabilities[:, 2])
    mask = (confidence > threshold) & (y.to_numpy() != 1)
    predicted = (probabilities[mask, 2] > probabilities[mask, 0]).astype("int64")
    actual = (y.to_numpy()[mask] == 2).astype("int64")
    return macro_f1(predicted, actual)


def evaluate_calibrated(
    model: CalibratedModel, frame: pd.DataFrame, start: datetime, end: datetime, flat_band: float
) -> dict[str, Any]:
    dataset = make_dataset(frame, start, end, flat_band)
    return diagnostics(dataset.y, model.predict_proba(dataset.x))
