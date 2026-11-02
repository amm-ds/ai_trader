"""Read-only parsers for v11 paper-node text and JSONL outputs."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP = re.compile(r"(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
CONF = re.compile(r"(LONG|SHORT) conf=([0-9.]+) close=([0-9.]+)")
META = re.compile(r"SKIP META prob=([^ ]+) < ([0-9.]+)")
REGIME = re.compile(r"SKIP (LONG|SHORT) (uptrend|downtrend) regime=([-0-9.]+)")
CHOP = re.compile(r"SKIP CHOP te=([0-9.]+) < ([0-9.]+)")
COST = re.compile(r"SKIP COST conf=([0-9.]+) expected=([-0-9.]+) < ([0-9.]+)")
FILL = re.compile(r"last_qty=([0-9.]+), last_px=([0-9.]+) USDT, commission=([0-9.]+) USDT")


def parse_timestamp(text: str) -> datetime | None:
    match = TIMESTAMP.search(text)
    return datetime.fromisoformat(match.group(1).removesuffix("Z") + "+00:00") if match else None


def parse_text(path: Path, since: datetime | None = None) -> list[dict[str, Any]]:
    """Parse evidenced trades, vetoes, incidents, and kill events from one log."""

    events: list[dict[str, Any]] = []
    with path.open(errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = ANSI.sub("", raw).strip()
            timestamp = parse_timestamp(line)
            if timestamp is None or (since is not None and timestamp < since):
                continue
            event = _event(path, line_number, timestamp, line)
            if event is not None:
                events.append(event)
    return events


def _event(path: Path, line: int, timestamp: datetime, text: str) -> dict[str, Any] | None:
    base = {"event_id": f"{path.name}:{line}", "timestamp": timestamp.isoformat(), "source": str(path), "raw": text}
    match = CONF.search(text)
    if match:
        side, confidence, close = match.groups()
        return {**base, "kind": "taken", "side": side, "confidence": float(confidence), "price": float(close)}
    match = META.search(text)
    if match:
        probability, threshold = match.groups()
        return {**base, "kind": "blocked_meta", "probability": _number_or_text(probability), "threshold": float(threshold), "reason": "meta-gate"}
    match = COST.search(text)
    if match:
        confidence, expected, threshold = match.groups()
        return {**base, "kind": "blocked_cost", "confidence": float(confidence), "expected_edge_bps": float(expected), "cost_threshold_bps": float(threshold), "reason": "fee-negative-cost-gate"}
    match = REGIME.search(text)
    if match:
        side, regime, value = match.groups()
        return {**base, "kind": "blocked_regime", "side": side, "regime": regime, "value": float(value), "reason": "regime-filter"}
    match = CHOP.search(text)
    if match:
        efficiency, threshold = match.groups()
        return {**base, "kind": "blocked_chop", "efficiency": float(efficiency), "threshold": float(threshold), "reason": "chop-filter"}
    lowered = text.lower()
    if "no space left on device" in lowered:
        return {**base, "kind": "stale_data", "reason": "disk-full"}
    if "pong deadline expired" in lowered:
        return {**base, "kind": "stale_data", "reason": "websocket-timeout"}
    if "rest bar poll error" in lowered or "returned non-zero exit status" in lowered or "calledprocesserror" in lowered:
        return {**base, "kind": "stale_data", "reason": "feature-refresh-failure"}
    if re.search(r"kill.?switch|halted", lowered):
        return {**base, "kind": "kill_switch", "reason": "kill-switch-event"}
    if "detected updated features; reloading" in lowered:
        return {**base, "kind": "feature_refresh", "reason": "feature-reload"}
    return None


def parse_jsonl(path: Path, since: datetime | None = None) -> list[dict[str, Any]]:
    """Parse order/fill state, skipping malformed lines and unrelated snapshots."""

    events = []
    with path.open(errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            timestamp = datetime.fromtimestamp(float(value.get("ts", 0)), tz=UTC)
            if since is not None and timestamp < since:
                continue
            if value.get("type") not in ("order_submitted", "fill"):
                continue
            side = str(value.get("side", "UNKNOWN"))
            events.append(
                {
                    "event_id": f"{path.name}:{line_number}",
                    "timestamp": timestamp.isoformat(),
                    "source": str(path),
                    "kind": "fill" if value.get("type") == "fill" else "order",
                    "side": side,
                    "size": float(value.get("qty") or 0.0),
                    "price": float(value.get("price") or 0.0),
                    "raw": raw.strip(),
                }
            )
    return events


def latest_jsonl(log_dir: Path) -> Path | None:
    files = sorted(log_dir.glob("paper_2*.jsonl"))
    return files[-1] if files else None


def load_v11_events(repo: Path, since: datetime | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Read only known v11 output files and return normalized events plus paths."""

    results = repo / "results"
    paths = [results / "paper_v11_edge.out", results / "paper_v11_edge.err", results / "refresh_v11_features_cron.log"]
    state = latest_jsonl(repo / "logs")
    if state is not None:
        paths.append(state)
    events = []
    for path in paths:
        if not path.exists():
            continue
        events.extend(parse_jsonl(path, since) if path.suffix == ".jsonl" else parse_text(path, since))
    events.sort(key=lambda event: (event["timestamp"], event["event_id"]))
    return events, [str(path) for path in paths if path.exists()]


def _number_or_text(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value
