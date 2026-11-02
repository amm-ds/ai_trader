from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_trader.execution.paper import PaperBroker

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def open_long(broker: PaperBroker):
    return broker.open_position("BTC_USDT", "long", 2.0, 100.0, 99.0, NOW, "test", "crypto")


def test_long_pnl_includes_slippage_and_twelve_bps_round_trip(tmp_path: Path) -> None:
    broker = PaperBroker(tmp_path / "state.json")
    position = open_long(broker)
    assert position.entry_price == pytest.approx(100.05)
    assert position.initial_risk == pytest.approx(2.4378006)
    trade = broker.close_position("BTC_USDT", 110.0, NOW + timedelta(minutes=5), "target")
    assert trade.exit_price == pytest.approx(109.945)
    assert trade.gross_pnl == pytest.approx(19.79)
    assert trade.fees == pytest.approx(0.251994)
    assert trade.net_pnl == pytest.approx(19.538006)
    assert trade.r_multiple == pytest.approx(19.538006 / 2.4378006)
    assert broker.report()["closed_trades"] == 1


def test_short_pnl_math(tmp_path: Path) -> None:
    broker = PaperBroker(tmp_path / "state.json")
    broker.open_position("ETH_USDT", "short", 1.0, 100.0, 101.0, NOW, "test", "crypto")
    trade = broker.close_position("ETH_USDT", 90.0, NOW + timedelta(minutes=5))
    assert trade.gross_pnl == pytest.approx(9.905)
    assert trade.net_pnl == pytest.approx(9.791003)


def test_stop_auto_close_and_state_persistence(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    broker = PaperBroker(state)
    open_long(broker)
    restored = PaperBroker(state)
    assert len(restored.positions) == 1
    assert restored.process_bar("BTC_USDT", 101.0, 99.1, NOW + timedelta(minutes=1)) is None
    trade = restored.process_bar("BTC_USDT", 101.0, 99.0, NOW + timedelta(minutes=2))
    assert trade is not None
    assert trade.close_reason == "stop"
    assert trade.exit_price == pytest.approx(98.9505)
    assert trade.r_multiple == pytest.approx(-1.0)
    assert not restored.positions
    assert PaperBroker(state).closed_trades[0].net_pnl == trade.net_pnl


def test_target_auto_close_resolves_same_bar_stop_first(tmp_path: Path) -> None:
    broker = PaperBroker(tmp_path / "state.json")
    broker.open_position(
        "BTC_USDT", "long", 1.0, 100.0, 99.0, NOW, "test", "crypto", target_price=101.0
    )
    trade = broker.process_bar("BTC_USDT", 102.0, 100.0, NOW + timedelta(minutes=1))
    assert trade is not None and trade.close_reason == "target"
    broker.open_position(
        "BTC_USDT", "long", 1.0, 100.0, 99.0, NOW, "test", "crypto", target_price=101.0
    )
    ambiguous = broker.process_bar("BTC_USDT", 102.0, 98.0, NOW + timedelta(minutes=2))
    assert ambiguous is not None and ambiguous.close_reason == "stop"
