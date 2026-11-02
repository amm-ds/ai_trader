from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.models.mean_reversion import MeanReversionStrategy
from ai_trader.models.momentum import MomentumStrategy
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import TradeSignal
from ai_trader.signal.ensemble import EnsembleStrategy

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def no_flow() -> FlowWindow:
    return FlowWindow(NOW - timedelta(minutes=15), NOW, 0, 0.0, 0.0)


def regime_frames(trends: list[str], bb_pct: float = 0.5, vol_pct: float = 0.5):
    frames = {}
    for timeframe, trend in zip(("5m", "15m", "1h", "4h"), trends, strict=True):
        frames[timeframe] = pd.DataFrame(
            {"trend": [trend], "bb_width_pct": [bb_pct], "realized_vol_pct": [vol_pct]},
            index=pd.DatetimeIndex([NOW]),
        )
    return frames


def test_regime_classifier_covers_all_four_states() -> None:
    classifier = RegimeClassifier()
    assert classifier.evaluate(regime_frames(["bull"] * 4), NOW).label == "TREND"
    assert classifier.evaluate(regime_frames(["neutral"] * 4), NOW).label == "RANGE"
    assert classifier.evaluate(regime_frames(["bull", "bull", "neutral", "bear"]), NOW).label == "TRANSITION"
    assert classifier.evaluate(regime_frames(["bull"] * 4, vol_pct=0.96), NOW).label == "VOLATILE"


def momentum_frames(long: bool = True, expanding: bool = True):
    sign = 1 if long else -1
    widths = [0.1, 0.2 if expanding else 0.1]
    trend = "bull" if long else "bear"
    return {
        "5m": pd.DataFrame(
            {"c": [100.0, 100.0], "bb_width": widths, "atr14": [2.0, 2.0], "trend": [trend, trend]},
            index=pd.date_range(end=NOW, periods=2, freq="5min"),
        ),
        "15m": pd.DataFrame({"trend": [trend]}, index=pd.DatetimeIndex([NOW])),
        "1h": pd.DataFrame({"roc_1": [sign * 0.01], "trend": [trend]}, index=pd.DatetimeIndex([NOW])),
        "4h": pd.DataFrame({"roc_1": [sign * 0.02], "trend": [trend]}, index=pd.DatetimeIndex([NOW])),
    }


def test_momentum_long_and_short_mirrors() -> None:
    long = MomentumStrategy().evaluate("BTC", momentum_frames(), NOW, no_flow())
    short = MomentumStrategy().evaluate("BTC", momentum_frames(long=False), NOW, no_flow())
    assert long is not None and long.side == "long" and long.stop_price < long.entry_price
    assert short is not None and short.side == "short" and short.stop_price > short.entry_price
    assert MomentumStrategy().evaluate("BTC", momentum_frames(expanding=False), NOW, no_flow()) is None


def mean_reversion_frame(rsi: float, close: float, middle: float):
    return {
        "5m": pd.DataFrame(
            {
                "c": [close, close],
                "h": [close + 1, close + 1],
                "l": [close - 1, close - 1],
                "rsi3": [50.0, rsi],
                "bb_mid": [middle, middle],
                "atr14": [2.0, 2.0],
            },
            index=pd.date_range(end=NOW, periods=2, freq="5min"),
        ),
        "15m": pd.DataFrame(
            {"trend": ["bull" if rsi < 50 else "bear"]}, index=pd.DatetimeIndex([NOW])
        ),
    }


def test_mean_reversion_has_bb_mid_target_and_structural_stop() -> None:
    long = MeanReversionStrategy().evaluate("BTC", mean_reversion_frame(10, 90, 100), NOW, no_flow())
    short = MeanReversionStrategy().evaluate("BTC", mean_reversion_frame(90, 110, 100), NOW, no_flow())
    assert long is not None and long.side == "long" and long.target_price == 100
    assert long.stop_price < long.entry_price
    assert short is not None and short.side == "short" and short.target_price == 100
    assert short.stop_price > short.entry_price


def routed_signal() -> TradeSignal:
    return TradeSignal("BTC", "long", 100, 98, "test", "crypto", "bull", True, "test", ())


def test_trend_router_uses_band_then_momentum_and_never_mean_reversion() -> None:
    ensemble = EnsembleStrategy()
    ensemble.regime.evaluate = Mock(return_value=RegimeState("TREND", "bull", 4, 0.5, 0.5))
    ensemble.band_rejection.evaluate = Mock(return_value=None)
    ensemble.momentum.evaluate = Mock(return_value=routed_signal())
    ensemble.mean_reversion.evaluate = Mock()
    signal = ensemble.evaluate("BTC", {}, NOW, no_flow())
    assert signal is not None
    assert ensemble.last_strategy == "momentum"
    ensemble.mean_reversion.evaluate.assert_not_called()


def test_band_only_router_never_calls_momentum() -> None:
    ensemble = EnsembleStrategy(enable_momentum=False)
    ensemble.regime.evaluate = Mock(return_value=RegimeState("TREND", "bull", 4, 0.5, 0.5))
    ensemble.band_rejection.evaluate = Mock(return_value=None)
    ensemble.momentum.evaluate = Mock(return_value=routed_signal())
    assert ensemble.evaluate("BTC", {}, NOW, no_flow()) is None
    assert ensemble.last_strategy == "band_rejection"
    ensemble.momentum.evaluate.assert_not_called()


def test_range_router_uses_only_mean_reversion() -> None:
    ensemble = EnsembleStrategy()
    ensemble.regime.evaluate = Mock(return_value=RegimeState("RANGE", "neutral", 0, 0.5, 0.5))
    ensemble.mean_reversion.evaluate = Mock(return_value=routed_signal())
    ensemble.band_rejection.evaluate = Mock()
    ensemble.momentum.evaluate = Mock()
    assert ensemble.evaluate("BTC", {}, NOW, no_flow()) is not None
    ensemble.band_rejection.evaluate.assert_not_called()
    ensemble.momentum.evaluate.assert_not_called()


def test_volatile_and_transition_regimes_stand_down() -> None:
    for label in ("VOLATILE", "TRANSITION"):
        ensemble = EnsembleStrategy()
        ensemble.regime.evaluate = Mock(return_value=RegimeState(label, "neutral", 0, 0.9, 0.9))
        ensemble.band_rejection.evaluate = Mock()
        ensemble.momentum.evaluate = Mock()
        ensemble.mean_reversion.evaluate = Mock()
        assert ensemble.evaluate("BTC", {}, NOW, no_flow()) is None
        assert "stand-down" in ensemble.last_reason
        ensemble.band_rejection.evaluate.assert_not_called()
        ensemble.momentum.evaluate.assert_not_called()
        ensemble.mean_reversion.evaluate.assert_not_called()
