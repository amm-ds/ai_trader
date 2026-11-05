"""Kimi API client: OpenAI-compatible chat with strict JSON, cache, caps, and costs.

The API key lives only in the repo-root `.env` (gitignored, chmod 600) or the
`KIMI_API_KEY` environment variable. It is never printed, logged, committed, or
sent anywhere except the `Authorization` header of an HTTPS request to the
configured base URL.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_trader.agents.local_llm import DecisionSnapshot, LLMDecision

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_TIMEOUT_S = 120.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
DAILY_CALL_CAP = 600
PROMPT_VERSION = "kimi-001-prompt-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"
PRICE_PATH = REPO_ROOT / "analysis" / "data" / "usage_prices.json"
CAP_STATE_PATH = REPO_ROOT / "analysis" / "data" / "state" / "kimi_001_calls.json"

PROPOSE_KEYS = {"action", "entry", "stop", "tp", "confidence", "rationale", "uncertainty"}
_CALL_LOCK = threading.Lock()


class DailyCapReached(RuntimeError):
    """The preregistered per-UTC-day call cap was reached; resume another day."""


class KimiAuthError(RuntimeError):
    """No API key available in the environment or the repo `.env`."""


@dataclass(frozen=True, slots=True)
class KimiProposal:
    """One validated LLM proposal with usage and fallback diagnostics."""

    action: str
    entry: float | None
    stop: float | None
    tp: float | None
    confidence: float
    rationale: str
    uncertainty: str
    model: str
    fallback: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    usd: float


def load_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse simple KEY=VALUE lines; values are never logged by callers."""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def api_key() -> str:
    """Resolve the key from the process environment or the repo `.env` only."""

    env_key = os.environ.get("KIMI_API_KEY")
    if env_key:
        return env_key
    file_key = load_env_file().get("KIMI_API_KEY")
    if file_key and file_key != "replace_me":
        return file_key
    raise KimiAuthError("KIMI_API_KEY missing: set it or add it to the gitignored .env")


def resolve_model() -> str:
    """Model from the environment, then `.env`, then the preregistered default."""

    return os.environ.get("KIMI_MODEL") or load_env_file().get("KIMI_MODEL") or DEFAULT_MODEL


def resolve_base_url() -> str:
    """Base URL from the environment, then `.env`, then the Moonshot default."""

    return os.environ.get("KIMI_BASE_URL") or load_env_file().get("KIMI_BASE_URL") or BASE_URL


def context_hash(context: dict[str, Any]) -> str:
    """Canonical content hash; independent of dict insertion order."""

    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def price_for(model: str, path: Path = PRICE_PATH) -> dict[str, float]:
    """Return USD-per-Mtok rates; longest matching prefix wins, `*` is the fallback."""

    table = json.loads(path.read_text(encoding="utf-8"))
    best_length = -1
    selected: dict[str, Any] | None = None
    for prefix, rates in table.items():
        matches = prefix == "*" or model == prefix or model.startswith(prefix)
        if matches and len(prefix) > best_length:
            best_length = len(prefix)
            selected = rates
    if selected is None:
        raise ValueError(f"no price entry for model {model!r}")
    return {
        "input_usd_per_mtok": float(selected["input_usd_per_mtok"]),
        "output_usd_per_mtok": float(selected["output_usd_per_mtok"]),
    }


def usage_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost of one call at the estimate-marked per-Mtok rates."""

    rates = price_for(model)
    return (
        prompt_tokens / 1_000_000.0 * rates["input_usd_per_mtok"]
        + completion_tokens / 1_000_000.0 * rates["output_usd_per_mtok"]
    )


class CallLedger:
    """Persisted per-UTC-day HTTP call count for the preregistered daily cap."""

    def __init__(self, path: Path = CAP_STATE_PATH, cap: int = DAILY_CALL_CAP) -> None:
        self.path = path
        self.cap = cap

    def _today(self) -> str:
        return datetime.now(UTC).date().isoformat()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"date": self._today(), "calls": 0}
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"date": self._today(), "calls": 0}
        if state.get("date") != self._today():
            return {"date": self._today(), "calls": 0}
        return {"calls": int(state.get("calls", 0)), "date": self._today()}

    def count(self) -> int:
        return int(self._read()["calls"])

    def remaining(self) -> int:
        return self.cap - self.count()

    def record(self, calls: int = 1) -> None:
        state = self._read()
        state["calls"] = int(state["calls"]) + calls
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def assert_available(self) -> None:
        if self.remaining() <= 0:
            raise DailyCapReached(
                f"daily call cap {self.cap} reached for {self._today()}; resume on a later UTC day"
            )


class DecisionCache:
    """Deterministic (snapshot_hash, model, prompt_version) -> proposal cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._rows[(row["snapshot_hash"], row["model"], row["prompt_version"])] = row

    def get(self, snapshot_hash: str, model: str, prompt_version: str) -> dict[str, Any] | None:
        return self._rows.get((snapshot_hash, model, prompt_version))

    def put(self, row: dict[str, Any]) -> None:
        key = (row["snapshot_hash"], row["model"], row["prompt_version"])
        if key in self._rows:
            return
        self._rows[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def __len__(self) -> int:
        return len(self._rows)


def parse_proposal(
    content: str,
    model: str,
    latency_s: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> KimiProposal:
    """Validate the exact seven-field schema; reject fences, extras, and bad types."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != PROPOSE_KEYS:
        raise ValueError("invalid proposal schema")
    action = value["action"]
    if action not in ("flat", "long", "short"):
        raise ValueError("invalid action")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("invalid confidence")
    if not isinstance(value["rationale"], str) or not isinstance(value["uncertainty"], str):
        raise ValueError("invalid rationale/uncertainty")
    # The reasoning model often writes slightly over the advertised limits;
    # truncate instead of rejecting so near-valid JSON is not forced to retry.
    rationale = value["rationale"][:200]
    uncertainty = value["uncertainty"][:100]
    entry, stop, tp = (_optional_price(value[key]) for key in ("entry", "stop", "tp"))
    if action == "flat":
        entry = stop = tp = None
    elif entry is None or stop is None or tp is None:
        raise ValueError("long/short require entry, stop, and tp")
    return KimiProposal(
        action=action,
        entry=entry,
        stop=stop,
        tp=tp,
        confidence=float(confidence),
        rationale=rationale,
        uncertainty=uncertainty,
        model=model,
        fallback=False,
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        usd=usage_usd(model, prompt_tokens, completion_tokens),
    )


def _optional_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid price")
    price = float(value)
    if not price > 0:
        raise ValueError("invalid price")
    return price


class KimiClient:
    """Minimal OpenAI-compatible chat client with retry, backoff, cache, and caps."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cache: DecisionCache | None = None,
        ledger: CallLedger | None = None,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.model = model or resolve_model()
        self.base_url = (base_url or resolve_base_url()).rstrip("/")
        self.timeout_s = timeout_s
        self.cache = cache
        self.ledger = ledger or CallLedger()
        self.max_attempts = max_attempts
        self._key = api_key()

    def decide(self, snapshot: DecisionSnapshot) -> LLMDecision:
        """Protocol-compatible action-only decision (arm C reviewer path)."""

        proposal = self.propose({"snapshot": snapshot.to_dict()}, prompt_kind="review")
        return LLMDecision(
            proposal.action,
            proposal.confidence,
            proposal.rationale,
            proposal.uncertainty,
            proposal.model,
            proposal.fallback,
            proposal.latency_s,
            proposal.completion_tokens,
            proposal.completion_tokens / proposal.latency_s if proposal.latency_s > 0 else 0.0,
        )

    def propose(
        self, context: dict[str, Any], prompt_kind: str = "propose"
    ) -> KimiProposal:
        """Return a cached or freshly fetched proposal for one frozen context."""

        digest = context_hash(context)
        version = PROMPT_VERSION if prompt_kind == "propose" else f"{PROMPT_VERSION}-{prompt_kind}"
        cached = self.cache.get(digest, self.model, version) if self.cache else None
        if cached is not None:
            decision = cached["decision"]
            usage = cached["usage"]
            return KimiProposal(
                action=decision["action"],
                entry=decision.get("entry"),
                stop=decision.get("stop"),
                tp=decision.get("tp"),
                confidence=decision["confidence"],
                rationale=decision["rationale"],
                uncertainty=decision["uncertainty"],
                model=self.model,
                fallback=decision["fallback"],
                latency_s=0.0,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                usd=usage_usd(self.model, usage["prompt_tokens"], usage["completion_tokens"]),
            )
        prompt = self._prompt(context, kind=prompt_kind)
        started = time.perf_counter()
        content, prompt_tokens, completion_tokens = self._chat_with_retry(prompt)
        try:
            proposal = parse_proposal(
                content, self.model, time.perf_counter() - started, prompt_tokens, completion_tokens
            )
        except (ValueError, json.JSONDecodeError):
            retry_content, retry_prompt, retry_completion = self._chat_with_retry(
                prompt + "\nReturn only one valid JSON object with exactly these keys: "
                "action, entry, stop, tp, confidence, rationale, uncertainty. "
                "Use null for entry/stop/tp when action is flat. No prose or fences."
            )
            prompt_tokens += retry_prompt
            completion_tokens += retry_completion
            try:
                proposal = parse_proposal(
                    retry_content,
                    self.model,
                    time.perf_counter() - started,
                    prompt_tokens,
                    completion_tokens,
                )
            except (ValueError, json.JSONDecodeError):
                proposal = KimiProposal(
                    action="flat",
                    entry=None,
                    stop=None,
                    tp=None,
                    confidence=0.0,
                    rationale="fallback: invalid JSON after one retry",
                    uncertainty="invalid-json",
                    model=self.model,
                    fallback=True,
                    latency_s=time.perf_counter() - started,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usd=usage_usd(self.model, prompt_tokens, completion_tokens),
                )
        if self.cache is not None:
            self.cache.put(
                {
                    "snapshot_hash": digest,
                    "model": self.model,
                    "prompt_version": version,
                    "created_utc": datetime.now(UTC).isoformat(),
                    "decision": {
                        "action": proposal.action,
                        "entry": proposal.entry,
                        "stop": proposal.stop,
                        "tp": proposal.tp,
                        "confidence": proposal.confidence,
                        "rationale": proposal.rationale,
                        "uncertainty": proposal.uncertainty,
                        "fallback": proposal.fallback,
                    },
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_s": round(proposal.latency_s, 6),
                        "usd": proposal.usd,
                    },
                }
            )
        return proposal

    def _prompt(self, context: dict[str, Any], kind: str) -> str:
        if kind == "review":
            instruction = (
                "You are a futures decision reviewer. Return exactly one JSON object with keys "
                "action (flat|long|short), confidence (0-1), rationale (<=200 chars), uncertainty (<=100 chars). "
                "Do not use markdown or extra keys. Evaluate evidence; do not invent size or overrides."
            )
        else:
            instruction = (
                "You propose BTC_USDT perpetual futures entries on completed 15-minute bars. "
                "Rules: never fight the higher-timeframe trend; order flow confirms, level reclaims trigger; "
                "the sovereign governor vetoes any proposal whose stop is outside 0.1%-5% of entry, and "
                "it sizes from the stop distance at 2% equity risk. Respect the context's governor limits. "
                "Return exactly one JSON object with keys action (flat|long|short), entry, stop, tp "
                "(positive numbers, or null when flat), confidence (0-1), rationale (<=200 chars), "
                "uncertainty (<=100 chars). No prose, no markdown fences."
            )
        return instruction + "\n" + json.dumps(context, default=str)

    def _chat_with_retry(self, prompt: str) -> tuple[str, int, int]:
        """One chat completion with capped exponential backoff on 429/5xx/timeout."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # kimi-for-coding (K2.8 preview) accepts only temperature 1; the
            # deterministic cache plus the sensitivity spot-check bound variance.
            "temperature": 1,
            # reasoning tokens count against max_tokens; 600 truncated the
            # reasoning phase and returned empty content (observed 2026-11-04)
            "max_tokens": 4096,
        }
        last_error = ""
        for attempt in range(self.max_attempts):
            self.ledger.assert_available()
            try:
                with _CALL_LOCK:
                    body = self._request(payload)
                self.ledger.record(1)
                usage = body.get("usage", {})
                content = body["choices"][0]["message"]["content"]
                return (
                    content,
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                )
            except DailyCapReached:
                raise
            except OSError as error:
                self.ledger.record(1)
                last_error = str(error)
                if isinstance(error, HTTPError):
                    if error.code == 429:
                        retry_after = error.headers.get("Retry-After") if error.headers else None
                        delay = min(float(retry_after), 60.0) if retry_after else _backoff_s(attempt)
                    elif 500 <= error.code < 600:
                        delay = _backoff_s(attempt)
                    else:
                        raise  # other HTTP statuses (400/401/404) are not retryable
                else:
                    delay = _backoff_s(attempt)  # timeouts and connection errors
                time.sleep(delay)
        raise RuntimeError(f"kimi chat failed after {self.max_attempts} attempts: {last_error}")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
                # the coding endpoint 403s (error 1010) the default urllib UA
                "User-Agent": "ai-trader-kimi-001/0.1",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            body: dict[str, Any] = json.loads(response.read())
        return body


def _backoff_s(attempt: int) -> float:
    """Capped exponential backoff: 2s, 4s, 8s, ..."""

    return min(BACKOFF_BASE_S * (2.0**attempt), 60.0)
