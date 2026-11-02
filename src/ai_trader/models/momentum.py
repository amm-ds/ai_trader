"""Long/short 4h and 1h continuation with Bollinger-width expansion."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.indicators import completed_history
from ai_trader.signal.band_rejection import Side, TradeSignal


class MomentumStrategy:
    """Trade continuation only when both higher-timeframe ROC signs agree."""

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        """Return a causal momentum signal, or `None` when expansion is absent."""

        rows: dict[str, pd.Series] = {}
        histories: dict[str, pd.DataFrame] = {}
        for timeframe in ("5m", "15m", "1h", "4h"):
            history = completed_history(features[timeframe], as_of)
            if len(history) < (2 if timeframe == "5m" else 1):
                return None
            histories[timeframe] = history
            rows[timeframe] = history.iloc[-1]
        execution = rows["5m"]
        roc_4h = float(rows["4h"].get("roc_1", 0.0))
        roc_1h = float(rows["1h"].get("roc_1", 0.0))
        width = float(execution.get("bb_width", 0.0))
        previous = histories["5m"].iloc[-2]
        previous_width = float(previous.get("bb_width", 0.0))
        values = (roc_4h, roc_1h, width, previous_width)
        if not all(pd.notna(value) for value in values) or width <= previous_width:
            return None
        side: Side | None = None
        if roc_4h > 0 and roc_1h > 0:
            side = "long"
        elif roc_4h < 0 and roc_1h < 0:
            side = "short"
        if side is None or not _flow_agrees(side, flow):
            return None
        trend = "bull" if side == "long" else "bear"
        if any(str(row["trend"]) != trend for row in rows.values()):
            return None
        close = float(execution["c"])
        atr = float(execution.get("atr14", 0.0))
        if not pd.notna(atr) or atr <= 0:
            return None
        stop = close - 1.5 * atr if side == "long" else close + 1.5 * atr
        return TradeSignal(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_price=stop,
            thesis=f"momentum-{side}-4h-1h-expansion",
            correlation_group="crypto-beta",
            trend="bull" if side == "long" else "bear",
            cascade_confirmed=True,
            written_trigger="4h and 1h ROC signs agree; 5m BB width expands",
            conditions=(
                f"roc-4h:{roc_4h:.8f}",
                f"roc-1h:{roc_1h:.8f}",
                f"bb-width:{width:.8f}>{previous_width:.8f}",
                f"flow-delta:{flow.delta:.8f}",
            ),
        )


def _flow_agrees(side: Side, flow: FlowWindow) -> bool:
    return flow.events == 0 or (side == "long" and flow.delta > 0) or (
        side == "short" and flow.delta < 0
    )
