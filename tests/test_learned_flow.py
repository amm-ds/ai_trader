from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.models.learned_flow import (
    FEATURE_COLUMNS,
    Dataset,
    LearnedFlowModel,
    calibrate_threshold,
    diagnostics,
    make_dataset,
    model_frame,
)
from ai_trader.signal.learned_flow import LearnedFlowStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def features(count=12):
    index = pd.date_range(NOW, periods=count, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100, 110, count), index=index)
    execution = pd.DataFrame(
        {
            "c": close, "o": close - 0.1, "h": close + 1, "l": close - 1,
            "ema9": close - 0.5, "bb_mid": close - 0.4, "roc_1": 0.001,
            "atr14": 1.0, "rsi3": 60.0, "bb_width_pct": 0.5,
            "realized_vol_pct": 0.5, "band_score": 4.0, "trend": "bull",
        },
        index=index,
    )
    flow = pd.DataFrame(0.0, index=index, columns=FEATURE_COLUMNS[:22])
    flow["delta_15m"] = 10.0
    frames = {"15m": execution, "flow": flow}
    for timeframe in ("5m", "1h", "4h"):
        frames[timeframe] = pd.DataFrame(
            {"roc_1": [0.001] * count, "trend": ["bull"] * count}, index=index
        )
    return frames


def test_dataset_is_pooled_causal_and_drops_unavailable_forward_labels() -> None:
    dataset = make_dataset({"BTC": features(), "ETH": features()}, NOW, NOW + timedelta(hours=3))
    assert len(dataset.x) == 18
    assert list(dataset.x.columns) == list(FEATURE_COLUMNS)
    assert set(dataset.y.unique()).issubset({0, 1, 2})
    frame = model_frame(features())
    assert frame.index[0] == NOW
    assert "close" in frame and "return_1h" in frame


def test_lightgbm_fit_predict_and_gain_importances() -> None:
    x = pd.DataFrame(
        {
            "signal": np.tile(np.linspace(-1, 1, 20), 3),
            "noise": np.arange(60),
        }
    )
    y = pd.Series(np.tile([0, 1, 2], 20))
    model = LearnedFlowModel(seed=42)
    model.fit(Dataset(x, y))
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (60, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert len(model.feature_importances()) == 2


def test_flat_excluded_auc_and_calibration() -> None:
    y = pd.Series([0, 2, 1, 0, 2])
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.1, 0.7]]
    )
    result = diagnostics(y, probabilities)
    assert result["accuracy"] == 1.0
    assert result["auc"] == 1.0
    assert result["flat_excluded_rows"] == 4
    assert calibrate_threshold(y, probabilities) >= 0.5


def test_learned_strategy_requires_confidence_and_aligned_15m_gate() -> None:
    frames = features()
    probabilities = pd.DataFrame(
        {"down": [0.1] * 12, "flat": [0.1] * 12, "up": [0.8] * 12},
        index=frames["15m"].index,
    )
    strategy = LearnedFlowStrategy({"BTC": probabilities}, 0.75)
    flow = FlowWindow(NOW, NOW + timedelta(minutes=15), 0, 0, 0)
    signal = strategy.evaluate("BTC", frames, NOW + timedelta(minutes=15), flow)
    assert signal is not None and signal.side == "long"
    assert signal.thesis == "learned-flow-long"
    assert LearnedFlowStrategy({"BTC": probabilities}, 0.9).evaluate(
        "BTC", frames, NOW + timedelta(minutes=15), flow
    ) is None
