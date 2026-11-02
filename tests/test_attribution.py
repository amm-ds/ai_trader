from datetime import UTC, datetime, timedelta

from ai_trader.evaluation.backtester import EntryAudit
from ai_trader.evaluation.metrics import performance_metrics
from ai_trader.execution.paper import PaperBroker

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_strategy_regime_attribution_contains_win_rate_and_pf_inputs() -> None:
    broker = PaperBroker("unused.json", persist=False)
    entries = []
    for index, (strategy, exit_price) in enumerate((("momentum", 90.0), ("momentum", 110.0), ("band_rejection", 105.0))):
        position = broker.open_position("BTC", "long", 1, 100, 99, NOW + timedelta(minutes=index), strategy, "crypto")
        broker.close_position("BTC", exit_price, NOW + timedelta(minutes=index + 1))
        entries.append(EntryAudit(position.opened_at, "BTC", "long", "bull", True, 1, position.position_id, "TREND", strategy))
    metrics = performance_metrics(broker.closed_trades, entries, NOW, NOW + timedelta(days=1))
    momentum = metrics["strategy_attribution"]["momentum"]["TREND"]
    assert momentum["trades"] == 2
    assert momentum["wins"] == 1
    assert momentum["gross_profit"] > 0
    assert momentum["gross_loss"] > 0
    assert metrics["strategy_attribution"]["band_rejection"]["TREND"]["trades"] == 1
