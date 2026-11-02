"""Calibrated microstructure signals with probability and mandatory cost gates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.models.micro_edge import ROUND_TRIP_COST
from ai_trader.models.regime import RegimeClassifier, RegimeState
from ai_trader.signal.band_rejection import Side, TradeSignal
from ai_trader.signal.exits import trend_target


class MicroEdgeStrategy:
    """Enter only when calibrated confidence and expected move exceed costs."""

    def __init__(self, probabilities: pd.DataFrame, threshold: float) -> None:
        self.probabilities = probabilities
        self.threshold = threshold
        self.regime = RegimeClassifier()
        self.last_regime: RegimeState | None = None
        self.last_strategy = "micro_edge"
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
        history = completed_history(self.probabilities, as_of)
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
        expected_move = float(row["expected_move_1h"])
        cost_gate = (2 * confidence - 1) * expected_move > ROUND_TRIP_COST
        if confidence <= self.threshold or not cost_gate or str(execution.iloc[-1]["trend"]) != trend:
            self.last_reason = "probability-cost-or-structure-gate"
            return None
        price = execution.iloc[-1]
        close = float(price["c"])
        buffer_ = max(0.1 * float(price["atr14"]), close * 0.001)
        stop = float(price["l"]) - buffer_ if side == "long" else float(price["h"]) + buffer_
        self.last_reason = "calibrated-cost-gated-signal"
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=stop,
            target_price=trend_target({"5m": execution, "1h": execution}, side, as_of),
            thesis=f"micro-edge-{side}",
            correlation_group="crypto-beta",
            trend=trend,
            cascade_confirmed=True,
            written_trigger="OOF-isotonic probability and expected-move cost gate with 15m alignment",
            conditions=(
                f"probability:{confidence:.6f}",
                f"expected-move:{expected_move:.8f}",
                f"round-trip-cost:{ROUND_TRIP_COST:.8f}",
            ),
        )
