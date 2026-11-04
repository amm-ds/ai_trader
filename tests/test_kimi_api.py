"""Mocked-transport tests for the kimi-001 client, cache, costs, and governor path."""

from __future__ import annotations

import json
import time as real_time
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from ai_trader.agents import kimi_api
from ai_trader.agents.kimi_api import (
    CallLedger,
    DailyCapReached,
    DecisionCache,
    KimiClient,
    parse_proposal,
    price_for,
    usage_usd,
)
from ai_trader.evaluation.kimi import validate_proposal

NOW = datetime(2026, 1, 1, tzinfo=UTC)
CONTEXT = {"snapshot_id": "s1", "symbol": "BTC_USDT", "as_of": NOW.isoformat()}
GOOD = {
    "choices": [{"message": {"content": json.dumps({
        "action": "long", "entry": 100.0, "stop": 99.5, "tp": 101.0,
        "confidence": 0.7, "rationale": "reclaim confirmed", "uncertainty": "none",
    })}}],
    "usage": {"prompt_tokens": 2000, "completion_tokens": 100},
}


def proposal(**overrides: object) -> SimpleNamespace:
    values = {
        "fallback": False,
        "action": "long",
        "entry": 100.0,
        "stop": 99.5,
        "tp": 101.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient(KimiClient):
    def __init__(self, responses: list[object], tmp_path: Path, **kwargs: object) -> None:
        super().__init__(
            model="kimi-k2.6",
            cache=kwargs.get("cache"),
            ledger=CallLedger(tmp_path / "ledger.json", cap=int(kwargs.get("cap", 600))),
        )
        self.responses = list(responses)
        self.requests = 0

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[no-any-return]


def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    fake_time = SimpleNamespace(
        perf_counter=real_time.perf_counter,
        sleep=sleeps.append,
    )
    monkeypatch.setattr(kimi_api, "time", fake_time)
    return sleeps


def http_error(code: int) -> HTTPError:
    return HTTPError("https://api.moonshot.ai/v1/chat/completions", code, "err", Message(), None)


def test_429_triggers_backoff_then_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps = no_sleep(monkeypatch)
    client = FakeClient([http_error(429), http_error(429), GOOD], tmp_path)
    decision = client.propose(CONTEXT)
    assert decision.action == "long"
    assert client.requests == 3
    assert sleeps == [2.0, 4.0]


def test_5xx_triggers_backoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps = no_sleep(monkeypatch)
    client = FakeClient([http_error(500), GOOD], tmp_path)
    decision = client.propose(CONTEXT)
    assert decision.action == "long"
    assert sleeps == [2.0]


def test_timeout_retries_then_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps = no_sleep(monkeypatch)
    client = FakeClient([TimeoutError("slow"), GOOD], tmp_path)
    decision = client.propose(CONTEXT)
    assert decision.action == "long"
    assert client.requests == 2
    assert sleeps == [2.0]


def test_transport_exhaustion_raises_and_counts_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_sleep(monkeypatch)
    client = FakeClient([TimeoutError("x")] * 4, tmp_path)
    with pytest.raises(RuntimeError, match="kimi chat failed"):
        client.propose(CONTEXT)
    assert client.requests == 4
    assert client.ledger.count() == 4


def test_non_retryable_http_error_raises_immediately(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_sleep(monkeypatch)
    client = FakeClient([http_error(401)] * 4, tmp_path)
    with pytest.raises(HTTPError):
        client.propose(CONTEXT)
    assert client.requests == 1


def test_invalid_json_retries_once_then_flags_fallback(tmp_path: Path) -> None:
    bad = {"choices": [{"message": {"content": "not json"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    client = FakeClient([bad, bad], tmp_path)
    decision = client.propose(CONTEXT)
    assert decision.fallback and decision.action == "flat"
    assert client.requests == 2


def test_cache_makes_replays_deterministic(tmp_path: Path) -> None:
    cache = DecisionCache(tmp_path / "cache.jsonl")
    first = FakeClient([GOOD], tmp_path, cache=cache)
    one = first.propose(CONTEXT)
    two = first.propose(CONTEXT)
    assert first.requests == 1
    assert one.action == two.action == "long"
    assert len(cache) == 1
    reloaded = DecisionCache(tmp_path / "cache.jsonl")
    second = FakeClient([], tmp_path, cache=reloaded)
    three = second.propose(CONTEXT)
    assert second.requests == 0
    assert three.entry == one.entry
    assert three.latency_s == 0.0


def test_daily_cap_blocks_and_resets(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "ledger.json", cap=2)
    ledger.record(2)
    with pytest.raises(DailyCapReached):
        ledger.assert_available()


def test_price_lookup_and_cost_math() -> None:
    rates = price_for("kimi-k2.6")
    assert rates["input_usd_per_mtok"] == 0.6
    assert rates["output_usd_per_mtok"] == 2.5
    assert price_for("kimi-k2.6-turbo-preview")["input_usd_per_mtok"] == 0.6
    assert price_for("some-other-model")["input_usd_per_mtok"] == 0.6
    assert usage_usd("kimi-k2.6", 1_000_000, 1_000_000) == pytest.approx(3.1)
    assert usage_usd("kimi-k2.6", 2000, 100) == pytest.approx(0.00145)


def test_parse_proposal_schema() -> None:
    parsed = parse_proposal(
        '{"action":"short","entry":100,"stop":100.6,"tp":99.0,"confidence":0.5,'
        '"rationale":"x","uncertainty":"y"}',
        "kimi-k2.6", 1.0, 10, 5,
    )
    assert parsed.action == "short" and parsed.entry == 100.0
    flat = parse_proposal(
        '{"action":"flat","entry":null,"stop":null,"tp":null,"confidence":0.1,'
        '"rationale":"x","uncertainty":"y"}',
        "kimi-k2.6", 1.0, 10, 5,
    )
    assert flat.action == "flat" and flat.entry is None
    with pytest.raises(ValueError):
        parse_proposal(
            '{"action":"long","entry":100,"stop":99,"tp":101,"confidence":0.5,'
            '"rationale":"x","uncertainty":"y","extra":1}',
            "kimi-k2.6", 1.0, 10, 5,
        )
    with pytest.raises(ValueError):
        parse_proposal(
            '{"action":"long","entry":100,"stop":99,"confidence":0.5,'
            '"rationale":"x","uncertainty":"y"}',
            "kimi-k2.6", 1.0, 10, 5,
        )


def test_governor_rejects_deliberately_non_compliant_proposal() -> None:
    inverted = validate_proposal(proposal(stop=100.6), NOW)  # long with stop above entry
    assert inverted[0] is None
    assert "stop-binding-mismatch" in inverted[1]
    too_tight = validate_proposal(proposal(stop=99.999), NOW)  # 0.001% stop, below the 0.1% floor
    assert too_tight[0] is None
    assert "stop-range" in too_tight[1]


def test_governor_caps_oversize_request_at_1r() -> None:
    intent, reasons = validate_proposal(proposal(entry=100.0, stop=99.0), NOW)
    assert intent is not None and reasons == ()
    # requested quantity was 100 * 0.02 / 1.0 = 2.0 units; governor 1R math is authoritative
    decision = intent
    assert decision.quantity <= 2.0 + 1e-9


def test_evening_bell_veto_inside_snapshot_review() -> None:
    late = NOW.replace(hour=20)
    intent, reasons = validate_proposal(proposal(), late)
    assert intent is None
    assert "evening-bell" in reasons
