from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ai_trader.risk.governor import (
    Governor,
    OrderIntent,
    PositionSnapshot,
    RiskContext,
)

NOW = datetime(2026, 10, 8, 12, tzinfo=UTC)


def intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTC_USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        stop_price=99.0,
        thesis="test",
        correlation_group="crypto",
        timestamp=NOW,
        data_timestamp=NOW,
        written_trigger="test trigger",
    )


def context() -> RiskContext:
    return RiskContext(now=NOW, equity=100.0)


@pytest.mark.parametrize(
    ("proposed", "state", "reason"),
    [
        (intent(), replace(context(), day_realized_r=-3.0), "daily-loss-halt"),
        (replace(intent(), data_timestamp=NOW + timedelta(seconds=1)), context(), "future-timestamp"),
        (replace(intent(), timestamp=NOW + timedelta(seconds=1)), context(), "future-timestamp"),
        (replace(intent(), data_timestamp=NOW - timedelta(minutes=16)), context(), "stale-data"),
        (intent(), replace(context(), now=NOW.replace(hour=19)), "evening-bell"),
        (
            intent(),
            replace(context(), positions=(PositionSnapshot("BTC_USDT", "other"),)),
            "no-averaging-down",
        ),
        (
            intent(),
            replace(
                context(),
                positions=tuple(PositionSnapshot(f"S{i}", f"g{i}") for i in range(3)),
            ),
            "position-cap",
        ),
        (
            intent(),
            replace(
                context(),
                positions=(PositionSnapshot("ETH", "crypto"), PositionSnapshot("SOL", "crypto")),
            ),
            "thesis-correlation-cap",
        ),
        (
            intent(),
            replace(context(), last_exit_at={"BTC_USDT": NOW - timedelta(hours=1)}),
            "cooldown",
        ),
        (
            intent(),
            replace(
                context(),
                last_exit_at={"BTC_USDT": NOW - timedelta(hours=5)},
                reentry_counts={"BTC_USDT": 1},
            ),
            "reentry-cap",
        ),
        (
            replace(intent(), written_trigger=None),
            replace(context(), last_exit_at={"BTC_USDT": NOW - timedelta(hours=5)}),
            "reentry-trigger-required",
        ),
        (replace(intent(), stop_price=101.0), context(), "stop-binding-mismatch"),
        (replace(intent(), stop_price=90.0), context(), "stop-range"),
        (replace(intent(), stop_price=99.9995), context(), "stop-range"),
        (intent(), replace(context(), kill_switch="operator"), "kill-switch:operator"),
        (intent(), replace(context(), halted=True), "global-halt"),
        (replace(intent(), observed_slippage_bps=11.0), context(), "slippage-kill-switch"),
        (intent(), replace(context(), equity=0.0), "invalid-equity"),
    ],
)
def test_each_veto_independently(proposed: OrderIntent, state: RiskContext, reason: str) -> None:
    decision = Governor().review(proposed, state)
    assert not decision.allowed
    assert decision.approved_quantity == 0.0
    assert reason in decision.reasons


def test_governor_shrinks_to_one_r_and_never_increases() -> None:
    large = Governor().review(replace(intent(), quantity=10.0), context())
    assert large.allowed
    assert large.approved_quantity == pytest.approx(2.0 / 1.2189003)
    assert large.adjustments == ("size-reduced-to-1r",)
    small = Governor().review(replace(intent(), quantity=0.25), context())
    assert small.allowed
    assert small.approved_quantity == 0.25
    assert small.approved_quantity <= intent().quantity


def test_naive_timestamp_fails_closed_instead_of_raising() -> None:
    naive = NOW.replace(tzinfo=None)
    decision = Governor().review(replace(intent(), data_timestamp=naive), context())
    assert not decision.allowed
    assert decision.reasons == ("timestamp-invalid",)
