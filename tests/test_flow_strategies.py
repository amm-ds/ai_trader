from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.signal.flow_strategies import FlowEnsembleStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def no_flow():
    return FlowWindow(NOW - timedelta(minutes=15), NOW, 0, 0, 0)


def base_features(trend="bull"):
    index = pd.date_range(NOW - timedelta(minutes=15), NOW, freq="15min", tz="UTC")
    execution = pd.DataFrame(
        {
            "o": [100, 100.5], "h": [101, 102], "l": [99, 100],
            "c": [100, 101], "ema9": [100, 100], "bb_mid": [99, 99],
            "atr14": [1, 1], "trend": ["neutral" if trend == "bull" else "neutral", trend],
            "bb_width_pct": [0.5, 0.5], "realized_vol_pct": [0.5, 0.5],
        },
        index=index,
    )
    frames = {"15m": execution}
    for timeframe in ("5m", "1h", "4h"):
        frames[timeframe] = pd.DataFrame(
            {
                "trend": ["neutral"], "bb_mid": [100.0], "ma60": [100.0],
                "bb_width_pct": [0.5], "realized_vol_pct": [0.5],
            },
            index=pd.DatetimeIndex([NOW]),
        )
    return frames


def flow_rows(log_odds=2.5, delta=20, previous_delta=10, cvd=5, percentile=0.8):
    return pd.DataFrame(
        {
            "absorption_log_odds_2h": [log_odds, log_odds],
            "absorption_log_odds_6h": [1.0, 1.0],
            "delta_15m": [previous_delta, delta],
            "cvd_change_15m": [cvd, cvd],
            "cvd_slope_pct": [percentile, percentile],
        },
        index=pd.date_range(NOW - timedelta(minutes=15), NOW, freq="15min", tz="UTC"),
    )


def test_spring_requires_absorption_and_15m_reclaim() -> None:
    features = base_features("bull")
    features["flow"] = flow_rows()
    signal = FlowEnsembleStrategy(2.0).evaluate("BTC", features, NOW, no_flow())
    assert signal is not None and signal.side == "long"
    assert signal.thesis == "spring_reversal-long"
    features["15m"].iloc[0, features["15m"].columns.get_loc("trend")] = "bull"
    assert FlowEnsembleStrategy(2.0).evaluate("BTC", features, NOW, no_flow()) is None


def test_continuation_requires_resume_delta_expansion_and_cvd_quantile() -> None:
    features = base_features("bull")
    for timeframe in ("5m", "1h", "4h"):
        features[timeframe]["trend"] = "bull"
    features["flow"] = flow_rows()
    signal = FlowEnsembleStrategy(cvd_slope_quantile=0.75).evaluate("BTC", features, NOW, no_flow())
    assert signal is not None and signal.thesis == "flow_continuation-long"
    features["flow"] = flow_rows(percentile=0.7)
    assert FlowEnsembleStrategy(cvd_slope_quantile=0.75).evaluate("BTC", features, NOW, no_flow()) is None
