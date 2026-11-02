from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ai_trader.data.archive import PerSymbolView
from ai_trader.data.flow import FlowWindow
from ai_trader.evaluation.backtester import BacktestConfig, Backtester
from ai_trader.execution.paper import PaperBroker
from ai_trader.features.indicators import add_strategy_features
from ai_trader.models.regime import RegimeState
from ai_trader.signal.band_rejection import TradeSignal
from ai_trader.signal.exits import (
    chandelier_stop,
    profit_ratchet_armed,
    structural_break,
)
from tests.helpers import flow, klines

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_profit_ratchet_arms_at_one_price_r_for_both_sides() -> None:
    assert profit_ratchet_armed("long", 100, 99, 101, 95)
    assert not profit_ratchet_armed("long", 100, 99, 100.9, 95)
    assert profit_ratchet_armed("short", 100, 101, 105, 99)
    assert not profit_ratchet_armed("short", 100, 101, 105, 99.1)


def test_chandelier_ratchets_to_breakeven_and_never_widens() -> None:
    assert chandelier_stop("long", 100, 99, 105, 95, 1, False) == 99
    assert chandelier_stop("long", 100, 99, 105, 95, 1, True) == 102
    assert chandelier_stop("short", 100, 101, 105, 95, 1, True) == 98
    rng = np.random.default_rng(42)
    for side in ("long", "short"):
        current = 99.0 if side == "long" else 101.0
        for _ in range(1000):
            updated = chandelier_stop(
                side, 100, current, rng.uniform(100, 110), rng.uniform(90, 100), 1, True
            )
            assert updated >= current if side == "long" else updated <= current
            current = updated


def test_broker_rejects_stop_widening() -> None:
    broker = PaperBroker("unused.json", persist=False)
    broker.open_position("BTC", "long", 1, 100, 99, NOW, "test", "crypto")
    assert broker.ratchet_stop("BTC", 100).stop_price == 100
    with pytest.raises(ValueError, match="widening"):
        broker.ratchet_stop("BTC", 99.5)


def test_structural_break_requires_band_and_flow_agreement() -> None:
    row = pd.Series({"c": 99.0, "ema9": 100.0})
    agreeing = FlowWindow(NOW - timedelta(minutes=15), NOW, 1, 0.0, 1.0)
    disagreeing = FlowWindow(NOW - timedelta(minutes=15), NOW, 1, 1.0, 0.0)
    assert structural_break("long", row, agreeing)
    assert not structural_break("long", row, disagreeing)
    assert not structural_break("short", row, agreeing)


class TrendSignal:
    def __init__(self) -> None:
        self.last_strategy = "band_rejection"
        self.last_regime = RegimeState("TREND", "bull", 4, 0.5, 0.5)
        self.last_reason = "signal"

    def evaluate(self, symbol, features, as_of, flow_window):
        if as_of != NOW + timedelta(minutes=15):
            return None
        close = float(features["5m"].iloc[-1]["c"])
        return TradeSignal(symbol, "long", close, close - 1, "test", "crypto", "bull", True, "test", ())


def test_event_driven_structural_break_and_strategy_audit() -> None:
    bars = klines(30)
    view = PerSymbolView("BTC_USDT", bars, flow([]))
    end = NOW + timedelta(minutes=30)
    features = {
        timeframe: add_strategy_features(view.bars(timeframe, as_of=end))
        for timeframe in ("5m", "15m", "1h", "4h")
    }
    features["5m"]["ema9"] = features["5m"]["c"].expanding().mean()
    break_time = NOW + timedelta(minutes=20)
    features["5m"].loc[break_time, "ema9"] = features["5m"].loc[break_time, "c"] + 1
    result = Backtester(
        {"BTC_USDT": view}, strategy=TrendSignal(), features={"BTC_USDT": features}
    ).run(BacktestConfig(NOW, end, structural_exits=True))
    assert len(result.trades) == 1
    assert result.trades[0].close_reason == "structural-break"
    assert result.entries[0].strategy == "band_rejection"
    assert result.entries[0].regime == "TREND"


def test_swing_uses_15m_execution_and_ignores_structural_break() -> None:
    bars = klines(30)
    view = PerSymbolView("BTC_USDT", bars, flow([]))
    end = NOW + timedelta(minutes=30)
    features = {
        timeframe: add_strategy_features(view.bars(timeframe, as_of=end))
        for timeframe in ("5m", "15m", "1h", "4h")
    }
    features["15m"]["ema9"] = features["15m"]["c"].expanding().mean()
    break_time = NOW + timedelta(minutes=30)
    features["15m"].loc[break_time, "ema9"] = features["15m"].loc[break_time, "c"] + 1
    result = Backtester(
        {"BTC_USDT": view}, strategy=TrendSignal(), features={"BTC_USDT": features}
    ).run(
        BacktestConfig(
            NOW,
            end,
            execution_timeframe="15m",
            structural_exits=True,
            swing_exits=True,
        )
    )
    assert len(result.trades) == 1
    assert result.trades[0].close_reason == "session-end"
