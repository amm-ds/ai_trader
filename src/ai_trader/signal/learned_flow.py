"""Trade LightGBM probabilities with structural risk and a 15m direction gate."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import Side, TradeSignal
from ai_trader.signal.exits import trend_target


class LearnedFlowStrategy:
    """Enter only on calibrated directional confidence, never on flat predictions."""

    def __init__(self, probabilities: Mapping[str, pd.DataFrame], threshold: float) -> None:
        self.probabilities = probabilities
        self.threshold = threshold
        self.regime = RegimeClassifier()
        self.last_regime: RegimeState | None = None
        self.last_strategy = "learned_flow"
        self.last_reason = "not-evaluated"

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        state = self.regime.evaluate(features, as_of)
        self.last_regime = state
        if state.label == "VOLATILE":
            self.last_reason = "regime-volatile-stand-down"
            return None
        history = completed_history(self.probabilities[symbol], as_of)
        execution = completed_history(features["15m"], as_of)
        if history.empty or execution.empty or history.index[-1] != as_of:
            self.last_reason = "no-model-row"
            return None
        row = history.iloc[-1]
        up = float(row["up"])
        down = float(row["down"])
        side: Side = "long" if up >= down else "short"
        confidence = max(up, down)
        trend: Literal["bull", "bear"] = "bull" if side == "long" else "bear"
        if confidence <= self.threshold or str(execution.iloc[-1]["trend"]) != trend:
            self.last_reason = "probability-or-structure-gate"
            return None
        price_row = execution.iloc[-1]
        close = float(price_row["c"])
        atr = float(price_row["atr14"])
        buffer_ = max(0.1 * atr, close * 0.001)
        stop = float(price_row["l"]) - buffer_ if side == "long" else float(price_row["h"]) + buffer_
        self.last_reason = "model-signal"
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=stop,
            target_price=trend_target({"5m": execution, "1h": execution}, side, as_of),
            thesis=f"learned-flow-{side}",
            correlation_group="crypto-beta",
            trend=trend,
            cascade_confirmed=True,
            written_trigger="calibrated LightGBM direction and aligned 15m band gate",
            conditions=(
                f"probability-up:{up:.6f}",
                f"probability-down:{down:.6f}",
                f"threshold:{self.threshold:.2f}",
            ),
        )
