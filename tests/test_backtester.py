from datetime import UTC, datetime, timedelta

from ai_trader.data.archive import PerSymbolView
from ai_trader.evaluation.backtester import BacktestConfig, Backtester
from ai_trader.signal.band_rejection import TradeSignal
from tests.helpers import flow, klines

NOW = datetime(2026, 1, 1, tzinfo=UTC)


class OneSignalStrategy:
    def evaluate(self, symbol, features, as_of, flow_window):
        if as_of != NOW + timedelta(minutes=15):
            return None
        return TradeSignal(
            symbol=symbol,
            side="long",
            entry_price=100.14,
            stop_price=99.0,
            thesis="integration-test",
            correlation_group="crypto",
            trend="bull",
            cascade_confirmed=True,
            written_trigger="test",
            conditions=(),
        )


def test_event_driven_integration_uses_governor_broker_and_stops() -> None:
    bars = klines(30)
    bars.iloc[16, bars.columns.get_loc("l")] = 98.9
    view = PerSymbolView("BTC_USDT", bars, flow([]))
    result = Backtester({"BTC_USDT": view}, strategy=OneSignalStrategy()).run(
        BacktestConfig(NOW, NOW + timedelta(minutes=30))
    )
    assert len(result.entries) == 1
    assert len(result.trades) == 1
    assert result.trades[0].close_reason == "stop"
    assert result.entries[0].cascade_confirmed
    assert result.decisions[-1].action == "entry"
