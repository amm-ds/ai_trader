"""RSI(3) exhaustion with a Bollinger-mid target for non-trending regimes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Literal

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.signal.band_rejection import Side, TradeSignal


class MeanReversionStrategy:
    """Fade three-bar exhaustion; the ensemble vetoes this in strong trends."""

    def __init__(self, oversold: float = 20.0, overbought: float = 80.0) -> None:
        self.oversold = oversold
        self.overbought = overbought

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        """Return an RSI/BB-mid reversion entry with a structural stop."""

        history = completed_history(features["5m"], as_of)
        if len(history) < 2:
            return None
        row = history.iloc[-1]
        rsi = float(row.get("rsi3", 50.0))
        close = float(row["c"])
        middle = float(row.get("bb_mid", close))
        atr = float(row.get("atr14", 0.0))
        if not all(pd.notna(value) for value in (rsi, middle, atr)) or atr <= 0:
            return None
        side: Side | None = None
        if rsi <= self.oversold and close < middle:
            side = "long"
        elif rsi >= self.overbought and close > middle:
            side = "short"
        if side is None or not _flow_agrees(side, flow):
            return None
        trend: Literal["bull", "bear"] = "bull" if side == "long" else "bear"
        confirmation = completed_history(features["15m"], as_of)
        if confirmation.empty or str(confirmation.iloc[-1]["trend"]) != trend:
            return None
        buffer_ = max(0.5 * atr, close * 0.001)
        stop = float(row["l"]) - buffer_ if side == "long" else float(row["h"]) + buffer_
        if (side == "long" and middle <= close) or (side == "short" and middle >= close):
            return None
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=stop,
            target_price=middle,
            thesis=f"mean-reversion-{side}-rsi3-bb-mid",
            correlation_group="crypto-beta",
            trend=trend,
            cascade_confirmed=True,
            written_trigger="15m band confirms; RSI(3) exhaustion; target BB mid",
            conditions=(
                f"15m-trend:{trend}",
                f"rsi3:{rsi:.4f}",
                f"bb-mid:{middle:.8f}",
                f"flow-delta:{flow.delta:.8f}",
            ),
        )


def _flow_agrees(side: Side, flow: FlowWindow) -> bool:
    return flow.events == 0 or (side == "long" and flow.delta > 0) or (
        side == "short" and flow.delta < 0
    )
