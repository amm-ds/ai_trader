from datetime import UTC, datetime, timedelta

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.signal.band_rejection import BandRejectionStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def feature_frames(trend: str, side: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in ("5m", "15m", "1h", "4h"):
        index = pd.date_range(end=NOW, periods=60, freq="5min")
        frame = pd.DataFrame(
            {"o": 100.0, "h": 100.2, "l": 99.8, "c": 100.0, "ema9": 100.0},
            index=index,
        )
        frame["trend"] = trend
        if timeframe == "5m":
            if side == "long":
                frame.iloc[-1] = [99.9, 100.2, 99.9, 100.1, 100.0, trend]
            else:
                frame.iloc[-1] = [100.1, 100.1, 99.8, 99.9, 100.0, trend]
        frames[timeframe] = frame
    return frames


def flow_window() -> FlowWindow:
    return FlowWindow(NOW - timedelta(minutes=15), NOW, 1, 2.0, 1.0)


def test_with_trend_long_rejection_has_cascade_audit() -> None:
    signal = BandRejectionStrategy().evaluate("BTC_USDT", feature_frames("bull", "long"), NOW, flow_window())
    assert signal is not None
    assert signal.side == "long"
    assert signal.trend == "bull"
    assert signal.cascade_confirmed
    assert signal.stop_price < signal.entry_price


def test_with_trend_short_rejection_has_cascade_audit() -> None:
    signal = BandRejectionStrategy().evaluate("BTC_USDT", feature_frames("bear", "short"), NOW, flow_window())
    assert signal is not None
    assert signal.side == "short"
    assert signal.trend == "bear"
    assert signal.stop_price > signal.entry_price


def test_mixed_cascade_does_not_trade() -> None:
    frames = feature_frames("bull", "long")
    frames["1h"]["trend"] = "bear"
    assert BandRejectionStrategy().evaluate("BTC_USDT", frames, NOW, flow_window()) is None
