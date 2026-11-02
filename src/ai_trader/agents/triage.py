"""One-call daily triage over read-only v11 paper-node events."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from ai_trader.adapters.v11_logs import load_v11_events

MODEL = "qwen3.6:35b-a3b-q4_K_M"
DEFAULT_REPO = Path("/Users/alejandro/work/trading_bots_crypto")


class TriageClient(Protocol):
    def summarize(self, prompt: str) -> tuple[str, dict[str, Any]]: ...


def build_prompt(events: list[dict[str, Any]], since: str) -> str:
    """Build a deterministic, evidence-only daily triage prompt."""

    payload = {
        "window": since,
        "event_count": len(events),
        "instruction": (
            "Write one plain-language daily triage note with sections: Taken trades, Blocked candidates, "
            "Anomalies/refresh health, and No-events if applicable. Cite only event_ids present in the input. "
            "Explain fee-negative vs stale feature vs threshold/meta/regime/chop blocks only when evidenced. "
            "Do not invent numbers, trades, incidents, causes, or recommendations. Use event_count exactly "
            "when reporting a count; do not estimate cadence or totals. Keep each explanation brief "
            "and include evidence ids. If zero events, say so."
        ),
        "events": events,
    }
    return json.dumps(payload, sort_keys=True)


class OllamaTriageClient:
    """Small stdlib client for one batched triage call per day."""

    def __init__(self, model: str = MODEL, base_url: str = "http://localhost:11434/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def summarize(self, prompt: str) -> tuple[str, dict[str, Any]]:
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 500},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read())
        usage = body.get("usage")
        if usage is None and "eval_count" in body:
            usage = {"completion_tokens": body["eval_count"]}
        return body["choices"][0]["message"]["content"], usage or {"note": "unavailable"}


def triage(repo: Path, since_text: str, client: TriageClient) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[str]]:
    """Parse one window and produce one note, or a static zero-event note."""

    since = _since(since_text)
    events, paths = load_v11_events(repo, since)
    if not events:
        return f"# V11 daily triage — {since_text}\n\nNo events found in this window.", {"note": "no-llm-call"}, events, paths
    prompt = build_prompt(events, since_text)
    note, usage = client.summarize(prompt)
    return note, usage, events, paths


def _since(value: str) -> datetime | None:
    if value.lower() == "all":
        return None
    if not value.endswith("h"):
        raise ValueError("since must be hours like 24h or 'all'")
    hours = int(value[:-1])
    return datetime.now(UTC) - timedelta(hours=hours)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="24h")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args(argv)
    note, usage, events, paths = triage(args.repo, args.since, OllamaTriageClient())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(note + "\n", encoding="utf-8")
    print(note)
    print("\nToken usage:", json.dumps(usage, sort_keys=True))
    print("Events:", len(events))
    print("Files:", json.dumps(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
