"""Strict-JSON local Ollama decisions with one retry and a safe fallback."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """The complete frozen context presented to quant and local-LLM decisions."""

    snapshot_id: str
    symbol: str
    as_of: str
    regime: dict[str, Any]
    model_probability: dict[str, Any]
    flow_features: dict[str, Any]
    governor: dict[str, Any]
    position: dict[str, Any]
    quant: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DecisionSnapshot:
        required = {"snapshot_id", "symbol", "as_of", "regime", "model_probability", "flow_features", "governor", "position", "quant"}
        if set(value) != required:
            raise ValueError("snapshot schema mismatch")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class LLMDecision:
    """Validated model output plus latency and fallback diagnostics."""

    action: str
    confidence: float
    rationale: str
    uncertainty: str
    model: str
    fallback: bool
    latency_s: float
    tokens: int
    tokens_per_second: float


class DecisionClient(Protocol):
    def decide(self, snapshot: DecisionSnapshot) -> LLMDecision: ...


def parse_decision(content: str, model: str, latency_s: float, tokens: int) -> LLMDecision:
    """Parse the exact four-field schema; reject fences, extras, and bad types."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"action", "confidence", "rationale", "uncertainty"}:
        raise ValueError("invalid decision schema")
    if value["action"] not in ("flat", "long", "short"):
        raise ValueError("invalid action")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("invalid confidence")
    if not isinstance(value["rationale"], str) or len(value["rationale"]) > 200:
        raise ValueError("invalid rationale")
    if not isinstance(value["uncertainty"], str) or len(value["uncertainty"]) > 100:
        raise ValueError("invalid uncertainty")
    return LLMDecision(
        value["action"], float(confidence), value["rationale"], value["uncertainty"],
        model, False, latency_s, tokens, tokens / latency_s if latency_s > 0 else 0.0,
    )


class OllamaClient:
    """Minimal OpenAI-compatible client for a local Ollama daemon."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", timeout_s: float = 180.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def decide(self, snapshot: DecisionSnapshot) -> LLMDecision:
        prompt = self._prompt(snapshot)
        started = time.perf_counter()
        first = self._call(prompt)
        try:
            return parse_decision(first[0], self.model, time.perf_counter() - started, first[1])
        except (ValueError, json.JSONDecodeError):
            retry = self._call(
                prompt + "\nReturn only one valid JSON object with exactly four fields: action, confidence, rationale, uncertainty. No prose or fences."
            )
            try:
                return parse_decision(retry[0], self.model, time.perf_counter() - started, first[1] + retry[1])
            except (ValueError, json.JSONDecodeError):
                return LLMDecision(
                    "flat", 0.0, "fallback: invalid JSON after one retry", "invalid-json",
                    self.model, True, time.perf_counter() - started, first[1] + retry[1],
                    (first[1] + retry[1]) / max(time.perf_counter() - started, 1e-9),
                )

    def _prompt(self, snapshot: DecisionSnapshot) -> str:
        return (
            "You are a futures decision reviewer. Return exactly one JSON object with keys "
            "action (flat|long|short), confidence (0-1), rationale (<=200 chars), uncertainty (<=100 chars). "
            "Do not use markdown or extra keys. Evaluate evidence; do not invent size or overrides.\n"
            + json.dumps(snapshot.to_dict(), sort_keys=True)
        )

    def _call(self, prompt: str) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 180},
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read())
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        tokens = int(usage.get("completion_tokens", body.get("eval_count", 0)))
        return content, tokens
