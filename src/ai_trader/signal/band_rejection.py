"""A minimal with-trend band-rejection entry strategy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import pandas as pd

from ai_trader.data.flow import FlowWindow
from ai_trader.features.band import REQUIRED_BARS
from ai_trader.features.indicators import completed_history

Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """An auditable entry, structural stop, and confirmed trend cascade."""

    symbol: str
    side: Side
    entry_price: float
    stop_price: float
    thesis: str
    correlation_group: str
    trend: Literal["bull", "bear"]
    cascade_confirmed: bool
    written_trigger: str
    conditions: tuple[str, ...]
    target_price: float | None = None


class BandRejectionStrategy:
    """Trade 5m rejection only when 5m/15m/1h/4h band votes agree."""

    def __init__(self, require_flow_agreement: bool = False) -> None:
        self.require_flow_agreement = require_flow_agreement

    def evaluate(
        self,
        symbol: str,
        features: Mapping[str, pd.DataFrame],
        as_of: datetime,
        flow: FlowWindow,
    ) -> TradeSignal | None:
        """Return a signal using completed rows at or before `as_of`, else `None`."""

        rows: dict[str, pd.Series] = {}
        for timeframe in ("5m", "15m", "1h", "4h"):
            history = completed_history(features[timeframe], as_of)
            if len(history) < REQUIRED_BARS:
                return None
            rows[timeframe] = history.iloc[-1]
        trends = {timeframe: str(row["trend"]) for timeframe, row in rows.items()}
        if len(set(trends.values())) != 1 or trends["15m"] not in ("bull", "bear"):
            return None
        trend = trends["15m"]
        execution = rows["5m"]
        close = float(execution["c"])
        open_ = float(execution["o"])
        high = float(execution["h"])
        low = float(execution["l"])
        ema9 = float(execution["ema9"])
        cached_atr = execution.get("atr14")
        atr = (
            float(cached_atr)
            if cached_atr is not None and pd.notna(cached_atr)
            else _atr(completed_history(features["5m"], as_of))
        )
        buffer_ = max(atr * 0.1, close * 0.0015)
        flow_disagrees = self.require_flow_agreement and flow.events > 0 and (
            (trend == "bull" and flow.delta <= 0) or (trend == "bear" and flow.delta >= 0)
        )

        if trend == "bear":
            rejected = high >= ema9 and close < ema9 and close < open_
            if not rejected or flow_disagrees:
                return None
            return TradeSignal(
                symbol=symbol,
                side="short",
                entry_price=close,
                stop_price=high + buffer_,
                thesis="downtrend-rally-rejection",
                correlation_group="crypto-beta",
                trend="bear",
                cascade_confirmed=True,
                written_trigger="15m/1h/4h bear cascade; 5m EMA9 rally rejection",
                conditions=("all-timeframes-bear", "5m-rally-rejected", f"flow-delta:{flow.delta:.8f}"),
            )
        rejected = low <= ema9 and close > ema9 and close > open_
        if not rejected or flow_disagrees:
            return None
        return TradeSignal(
            symbol=symbol,
            side="long",
            entry_price=close,
            stop_price=low - buffer_,
            thesis="uptrend-dip-rejection",
            correlation_group="crypto-beta",
            trend="bull",
            cascade_confirmed=True,
            written_trigger="15m/1h/4h bull cascade; 5m EMA9 dip rejection",
            conditions=("all-timeframes-bull", "5m-dip-rejected", f"flow-delta:{flow.delta:.8f}"),
        )


def _atr(bars: pd.DataFrame) -> float:
    previous_close = bars["c"].shift(1)
    ranges = pd.concat(
        [
            bars["h"] - bars["l"],
            (bars["h"] - previous_close).abs(),
            (bars["l"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = ranges.rolling(14, min_periods=14).mean().iloc[-1]
    return float(value) if pd.notna(value) else 0.0
