from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.models.micro_edge import (
    FEATURE_COLUMNS,
    fit_calibrated,
    make_dataset,
    model_frame,
    purged_folds,
)
from ai_trader.signal.micro_edge import MicroEdgeStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def frame(count=400):
    index = pd.date_range(NOW, periods=count, freq="15min", tz="UTC")
    close = 100 + np.sin(np.arange(count) / 5) * 0.3 + np.arange(count) * 0.001
    result = pd.DataFrame(0.0, index=index, columns=FEATURE_COLUMNS)
    result["c"] = close
    result["o"] = close - 0.01
    result["h"] = close + 0.2
    result["l"] = close - 0.2
    result["ema9"] = close - 0.05
    result["bb_mid"] = close - 0.04
    result["atr14"] = 1.0
    result["rsi3"] = 50.0
    result["bb_width_pct"] = 0.5
    result["realized_vol_pct"] = 0.5
    result["band_score"] = 4.0
    result["trend"] = "bull"
    for timeframe in ("5m", "15m", "1h", "4h"):
        result[f"trend_{timeframe}"] = "bull"
    return result


def test_labels_respect_horizon_and_flat_band() -> None:
    data = make_dataset(frame(), NOW, NOW + timedelta(days=3), 0.001)
    assert len(data.x) > 0
    assert set(data.y.unique()) == {0, 1, 2}


def test_purged_folds_have_five_oos_segments_and_embargo() -> None:
    folds = purged_folds(NOW, NOW + timedelta(days=800))
    assert len(folds) == 5
    for fold in folds:
        assert fold.test_start - fold.train_end == timedelta(days=1)
        assert fold.test_start < fold.test_end


def test_oof_isotonic_probabilities_are_normalized() -> None:
    data = frame()
    model, y, probabilities = fit_calibrated(data, NOW, NOW + timedelta(days=5), 0.001, seed=42)
    assert probabilities.shape == (len(y), 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    predicted = model.predict_proba(model_frame(data).loc[:, list(FEATURE_COLUMNS)])
    assert predicted.shape == (400, 3)
    assert model.feature_importances()


def test_cost_gate_blocks_positive_probability_with_too_small_expected_move() -> None:
    data = frame(12)
    probabilities = pd.DataFrame(
        {
            "down": 0.05, "flat": 0.05, "up": 0.9, "expected_move_1h": 0.001,
        },
        index=data.index,
    )
    strategy = MicroEdgeStrategy(probabilities, 0.6)
    flow = FlowWindow(NOW, NOW + timedelta(minutes=15), 0, 0, 0)
    assert strategy.evaluate("BTC", {"15m": data, "5m": data, "1h": data, "4h": data}, NOW + timedelta(minutes=15), flow) is None
    probabilities["expected_move_1h"] = 0.01
    signal = strategy.evaluate("BTC", {"15m": data, "5m": data, "1h": data, "4h": data}, NOW + timedelta(minutes=15), flow)
    assert signal is not None and signal.side == "long"
