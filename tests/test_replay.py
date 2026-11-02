from dataclasses import replace
from datetime import UTC, datetime, timedelta

from ai_trader.evaluation.backtester import BacktestResult, EntryAudit
from ai_trader.evaluation.replay import _cooldown_violations
from ai_trader.execution.paper import PaperBroker

NOW = datetime(2026, 1, 1, 22, tzinfo=UTC)


def churn_result(second_entry: datetime) -> BacktestResult:
    broker = PaperBroker("unused.json", persist=False)
    broker.open_position("BTC", "long", 1.0, 100.0, 99.0, NOW, "test", "crypto")
    first = broker.close_position("BTC", 99.0, NOW + timedelta(hours=1), "stop")
    second = replace(
        first,
        position_id="second",
        opened_at=second_entry,
        closed_at=second_entry + timedelta(hours=1),
    )
    entries = (
        EntryAudit(NOW, "BTC", "long", "bull", True, 1.0, "first"),
        EntryAudit(second_entry, "BTC", "long", "bull", True, 1.0, "second"),
    )
    return BacktestResult({}, (first, second), entries, (), {})


def test_churn_check_resets_with_governor_at_utc_midnight() -> None:
    overnight = churn_result(datetime(2026, 1, 2, 1, tzinfo=UTC))
    same_day = churn_result(datetime(2026, 1, 1, 23, 30, tzinfo=UTC))
    assert _cooldown_violations(overnight) == 0
    assert _cooldown_violations(same_day) == 1
