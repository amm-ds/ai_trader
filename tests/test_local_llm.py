from datetime import UTC, datetime, timedelta

import pytest

from ai_trader.agents.local_llm import DecisionSnapshot, LLMDecision, OllamaClient, parse_decision
from ai_trader.data.archive import PerSymbolView
from ai_trader.evaluation.backtester import BacktestConfig, Backtester
from ai_trader.signal.band_rejection import TradeSignal
from ai_trader.signal.llm_veto import LLMVetoStrategy
from tests.helpers import flow, klines

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def snapshot() -> DecisionSnapshot:
    return DecisionSnapshot(
        "s1", "BTC_USDT", NOW.isoformat(), {"label": "TREND"},
        {"up": 0.8, "down": 0.1, "flat": 0.1}, {"ofi_15m": 0.2},
        {"risk_fraction": 0.02}, {"open": False}, {"action": "long"},
    )


def test_schema_validation_and_snapshot_round_trip() -> None:
    restored = DecisionSnapshot.from_dict(snapshot().to_dict())
    assert restored == snapshot()
    decision = parse_decision(
        '{"action":"long","confidence":0.8,"rationale":"reclaim","uncertainty":"sample"}',
        "mock", 1.0, 10,
    )
    assert decision.action == "long" and decision.tokens_per_second == 10
    with pytest.raises(ValueError):
        parse_decision('{"action":"long","confidence":0.8}', "mock", 1, 1)
    with pytest.raises(ValueError):
        parse_decision('{"action":"long","confidence":1.2,"rationale":"x","uncertainty":"y"}', "mock", 1, 1)


class InvalidClient(OllamaClient):
    def __init__(self) -> None:
        super().__init__("mock")
        self.calls = 0

    def _call(self, prompt: str):
        self.calls += 1
        return "not-json", 5


def test_invalid_json_retries_once_then_falls_back() -> None:
    client = InvalidClient()
    decision = client.decide(snapshot())
    assert decision.fallback
    assert decision.action == "flat"
    assert client.calls == 2


class CannedClient:
    def __init__(self, action: str, confidence: float) -> None:
        self.action = action
        self.confidence = confidence

    def decide(self, snapshot: DecisionSnapshot) -> LLMDecision:
        return LLMDecision(self.action, self.confidence, "ok", "none", "mock", False, 0.1, 1, 10)


class OneSignal:
    def evaluate(self, symbol, features, as_of, flow_window):
        if as_of != NOW + timedelta(minutes=15):
            return None
        close = float(features["15m"].iloc[-1]["c"])
        return TradeSignal(symbol, "long", close, close - 1, "test", "crypto", "bull", True, "test", ())


def run_with_client(client) -> tuple[float, int]:
    bars = klines(30)
    view = PerSymbolView("BTC_USDT", bars, flow([]))
    result = Backtester(
        {"BTC_USDT": view}, strategy=LLMVetoStrategy(OneSignal(), client)
    ).run(BacktestConfig(NOW, NOW + timedelta(minutes=30)))
    quantity = result.entries[0].quantity if result.entries else 0.0
    return quantity, len(result.trades)


def test_mock_llm_counterfactual_is_deterministic_and_bounded() -> None:
    first = run_with_client(CannedClient("long", 0.5))
    second = run_with_client(CannedClient("long", 0.5))
    assert first == second
    assert first[1] == 1
    vetoed = run_with_client(CannedClient("flat", 1.0))
    assert vetoed == (0.0, 0)
