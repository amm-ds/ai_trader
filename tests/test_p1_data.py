from datetime import UTC, datetime, timedelta

import pandas as pd
import pandas.testing as pdt

from ai_trader.data.archive import PerSymbolView, load_flow_from_records
from ai_trader.features.indicators import add_strategy_features, completed_history
from tests.helpers import klines

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_p1_indicators_do_not_change_when_future_bars_are_appended() -> None:
    bars = klines(300)
    truncated = add_strategy_features(bars.iloc[:-1])
    extended = add_strategy_features(bars)
    pdt.assert_frame_equal(truncated, extended.iloc[:-1])


def test_completed_history_is_right_closed() -> None:
    bars = klines(10)
    history = completed_history(bars, NOW + timedelta(minutes=5))
    assert history.index[-1] == NOW + timedelta(minutes=5)
    assert len(history) == 6


def test_five_minute_source_resamples_to_completed_fifteen_minute_bars() -> None:
    source = klines(3).iloc[::1].copy()
    source.index = pd.date_range(NOW, periods=3, freq="5min")
    view = PerSymbolView("BTC_USDT", source, load_flow_from_records(), source_timeframe="5m")
    bars = view.bars("15m")
    assert len(bars) == 1
    assert bars.index[0] == NOW + timedelta(minutes=15)
    assert bars.iloc[0]["v"] == 3000.0
    assert view.one_minute_bars(NOW, NOW + timedelta(minutes=15)).equals(source)
