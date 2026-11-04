"""Arm B: Kimi-proposed entries mechanically validated by the sovereign governor.

The LLM reviews the market on a fixed cadence and may only *propose* direction,
entry, stop, and target. A deterministic adapter translates the proposal into a
`TradeSignal` anchored to the current bar; the unchanged governor then vetoes or
shrinks it. No proposal path can raise size beyond governor math.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

import pandas as pd

from ai_trader.agents.kimi_api import KimiClient
from ai_trader.data.flow import FlowWindow
from ai_trader.evaluation.backtester import SignalStrategy
from ai_trader.features.indicators import completed_history
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import TradeSignal

ContextBuilder = Callable[[str, Mapping[str, pd.DataFrame], datetime], dict[str, Any] | None]

DEFAULT_CADENCE = timedelta(hours=4)


class KimiProposalStrategy(SignalStrategy):
    """Propose entries via a cached Kimi client at a fixed review cadence."""

    def __init__(
        self,
        client: KimiClient,
        context_builder: ContextBuilder,
        cadence: timedelta = DEFAULT_CADENCE,
    ) -> None:
        if cadence <= timedelta(0):
            raise ValueError("review cadence must be positive")
        self.client = client
        self.context_builder = context_builder
        self.cadence = cadence
        self.regime = RegimeClassifier()
        self.last_regime: RegimeState | None = None
        self.last_strategy = "kimi_proposal"
        self.last_reason = "not-evaluated"
        self._last_review: datetime | None = None
        self.reviews = 0

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        if self._last_review is not None and as_of - self._last_review < self.cadence:
            self.last_reason = "between-reviews"
            return None
        context = self.context_builder(symbol, features, as_of)
        self.last_regime = self.regime.evaluate(features, as_of)
        if context is None:
            self.last_reason = "no-context"
            return None
        self._last_review = as_of
        self.reviews += 1
        proposal = self.client.propose(context)
        if proposal.fallback:
            self.last_reason = "llm-invalid"
            return None
        if proposal.action == "flat":
            self.last_reason = "llm-flat"
            return None
        side: Literal["long", "short"] = "long" if proposal.action == "long" else "short"
        trend_side = "bull" if side == "long" else "bear"
        state = self.last_regime
        if state.label == "TREND" and state.direction != "neutral" and trend_side != state.direction:
            self.last_reason = "adapter-counter-trend"
            return None
        aligned = state.direction == trend_side and state.aligned_timeframes >= 2
        if not aligned:
            self.last_reason = "adapter-no-cascade"
            return None
        history = completed_history(features["15m"], as_of)
        if history.empty or history.index[-1] != as_of:
            self.last_reason = "no-bar"
            return None
        close = float(history.iloc[-1]["c"])
        if proposal.entry is None or proposal.stop is None or proposal.tp is None:
            self.last_reason = "llm-missing-prices"
            return None
        stop_ratio = proposal.stop / proposal.entry
        target_ratio = proposal.tp / proposal.entry
        self.last_reason = "llm-proposal"
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=close * stop_ratio,
            target_price=close * target_ratio,
            thesis=f"kimi-{side}",
            correlation_group="crypto-beta",
            trend="bull" if side == "long" else "bear",
            cascade_confirmed=True,
            written_trigger="Kimi proposal re-based to the completed 15m close; governor validates",
            conditions=(
                f"llm-confidence:{proposal.confidence:.4f}",
                f"llm-stop-ratio:{stop_ratio:.6f}",
                f"llm-target-ratio:{target_ratio:.6f}",
            ),
        )
