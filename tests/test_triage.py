from pathlib import Path

from ai_trader.adapters.v11_logs import load_v11_events, parse_jsonl, parse_text
from ai_trader.agents.triage import build_prompt, triage


def test_text_parser_normalizes_taken_block_stale_and_kill(tmp_path: Path) -> None:
    log = tmp_path / "paper_v11_edge.out"
    log.write_text(
        "\x1b[1m2026-11-02T10:00:00.000000000Z\x1b[0m V11Edge LONG conf=0.825 close=81051.00\n"
        "\x1b[1m2026-11-02T10:15:00.000000000Z\x1b[0m V11Edge SKIP META prob=0.25 < 0.95\n"
        "\x1b[1m2026-11-02T10:30:00.000000000Z\x1b[0m V11Edge SKIP COST conf=0.82 expected=9.5 < 13.0\n"
        "\x1b[1m2026-11-02T10:45:00.000000000Z\x1b[0m PaperLogger error: No space left on device\n"
        "\x1b[1m2026-11-02T11:00:00.000000000Z\x1b[0m KillSwitch triggered\n"
        "malformed line\n"
    )
    events = parse_text(log)
    assert [event["kind"] for event in events] == ["taken", "blocked_meta", "blocked_cost", "stale_data", "kill_switch"]
    assert events[0]["confidence"] == 0.825
    assert events[1]["threshold"] == 0.95
    assert events[2]["expected_edge_bps"] == 9.5
    assert events[2]["cost_threshold_bps"] == 13.0
    assert events[3]["reason"] == "disk-full"
    assert events[0]["event_id"].endswith(":1")


def test_jsonl_tolerates_malformed_and_snapshots(tmp_path: Path) -> None:
    log = tmp_path / "paper_test.jsonl"
    log.write_text(
        '{"type":"snapshot","ts":1793600000}\n'
        "not-json\n"
        '{"type":"fill","side":"BUY","qty":0.01,"price":80000,"ts":1793600000}\n'
    )
    events = parse_jsonl(log)
    assert len(events) == 1
    assert events[0]["kind"] == "fill"
    assert events[0]["size"] == 0.01


def test_prompt_is_deterministic() -> None:
    events, _ = load_v11_events(Path("/tmp/does-not-exist"))
    prompt_a = build_prompt(events, "24h")
    prompt_b = build_prompt(events, "24h")
    assert prompt_a == prompt_b
    assert '"events": []' in prompt_a


class CannedClient:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, prompt: str):
        self.calls += 1
        return "TRIAGE", {"completion_tokens": 7}


def test_triage_batches_once_and_zero_event_skips_llm(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "paper_v11_edge.out").write_text("2026-11-02T10:00:00Z V11Edge SHORT conf=0.81 close=80000.00\n")
    client = CannedClient()
    note, usage, events, _ = triage(tmp_path, "all", client)
    assert note == "TRIAGE"
    assert usage["completion_tokens"] == 7
    assert len(events) == 1
    assert client.calls == 1
    empty = tmp_path / "empty"
    note, usage, events, _ = triage(empty, "all", client)
    assert "No events found" in note
    assert not events
    assert usage["note"] == "no-llm-call"
    assert client.calls == 1
